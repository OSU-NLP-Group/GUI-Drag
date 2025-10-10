from __future__ import annotations
import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Optional imports
try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
        smart_resize as qwen_smart_resize,
    )
except Exception as exc:
    raise ImportError(
        "qwen_smart_resize must be available for the evaluation bundle"
    ) from exc

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

try:
    from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
    ANTHROPIC_AVAILABLE = True
except Exception:
    AnthropicBedrock = None
    AsyncAnthropicBedrock = None
    ANTHROPIC_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Constants
MAX_ATTEMPTS = 5
NUM_SECONDS_TO_SLEEP = 5

# Prompt templates
FN_CALL_TEMPLATE = """You are a helpful assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {width}x{height}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n* `type`: Type a string of text on the keyboard.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button.\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n* `right_click`: Click the right mouse button.\n* `middle_click`: Click the middle mouse button.\n* `double_click`: Double-click the left mouse button.\n* `scroll`: Performs a scroll of the mouse scroll wheel.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click_drag`, `action=left_click`, `action=right_click`, `action=double_click`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}}}}}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

## Note
- Use Chinese in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- Note that sentences usually end with a period, question mark, or exclamation mark. A sentence is not a line of text but rather a span of text properly delimited by punctuation.

## User Instruction
{instruction}
"""


@dataclass
class SizeInfo:
    original_size: Tuple[int, int]
    resized_size: Tuple[int, int]


# Utility functions
def encode_image(image: Image) -> str:
    """Encode PIL image to base64 string"""
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str




def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def uitar_smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 100 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> Tuple[int, int]:
    """
    UITAR-specific smart resize function
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor'
    2. Total pixels are within [min_pixels, max_pixels]  
    3. Aspect ratio is maintained as closely as possible
    """
    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_coordinates(coordinates: Tuple[int, int], size_original: Tuple[int, int], size_resized: Tuple[int, int]) -> Tuple[int, int]:
    """Resize coordinates from one resolution to another"""
    return (
        round(coordinates[0] * size_resized[0] / size_original[0]), 
        round(coordinates[1] * size_resized[1] / size_original[1])
    )


def extract_tool_calls(text: str) -> List[str]:
    """Extract tool calls from model response"""
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def process_drag_response(parsed_responses: List[str], size_info: SizeInfo) -> List[Dict[str, Any]]:
    """Process parsed responses and convert to drag format"""
    allowed_actions = {
        'key', 'type', 'mouse_move', 'left_click', 'left_click_drag', 
        'right_click', 'middle_click', 'double_click', 'scroll', 'wait', 'terminate'
    }
    
    if not parsed_responses:
        return []
    
    result = []
    i = 0
    
    while i < len(parsed_responses):
        try:
            current_action_data = json.loads(parsed_responses[i])
            current_action = current_action_data['arguments']['action']
            
            if current_action not in allowed_actions:
                print(f"Warning: Unsupported action type: {current_action}")
                i += 1
                continue
            
            # Handle left_click_drag specially
            if current_action == 'left_click_drag':
                if i == 0:
                    print("Warning: left_click_drag must have a previous action")
                    i += 1
                    continue
                
                prev_action_data = json.loads(parsed_responses[i-1])
                prev_action = prev_action_data['arguments']['action']
                
                if prev_action not in ['left_click', 'mouse_move']:
                    print(f"Warning: Action before left_click_drag should be left_click or mouse_move, got: {prev_action}")
                
                # Get start and end coordinates
                start_coordinate = prev_action_data['arguments']['coordinate']
                end_coordinate = current_action_data['arguments']['coordinate']
                
                # Resize coordinates
                start_coordinate = resize_coordinates(
                    start_coordinate, size_info.resized_size, size_info.original_size
                )
                end_coordinate = resize_coordinates(
                    end_coordinate, size_info.resized_size, size_info.original_size
                )
                
                # Remove previous action if it was added
                if result and result[-1]['action'] == prev_action:
                    result.pop()
                
                # Add drag action
                result.append({
                    'action': 'drag',
                    'coordinates': [start_coordinate, end_coordinate]
                })
                
            else:
                # Handle other actions
                action_result = {'action': current_action}
                
                # Process coordinates if present
                if 'coordinate' in current_action_data['arguments']:
                    coordinate = current_action_data['arguments']['coordinate']
                    resized_coordinate = resize_coordinates(
                        coordinate, size_info.resized_size, size_info.original_size
                    )
                    action_result['coordinate'] = resized_coordinate
                
                # Keep other parameters
                for key, value in current_action_data['arguments'].items():
                    if key not in ['action', 'coordinate']:
                        action_result[key] = value
                
                result.append(action_result)
            
            i += 1
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response {i}: {e}")
            i += 1
            continue
    
    return result


def process_simple_drag_response(parsed_responses: List[str], size_info: SizeInfo) -> List[Dict[str, Any]]:
    """
    Process parsed responses for simple drag pattern only:
    - Takes only the first two actions from parsed_responses
    - First action: mouse_move or left_click
    - Second action: left_click_drag
    Returns empty list for any other pattern or parsing errors.
    """
    # Must have at least 2 actions
    if len(parsed_responses) < 2:
        return []
    
    try:
        # Only look at first two actions
        first_action_data = json.loads(parsed_responses[0])
        first_action = first_action_data['arguments']['action']
        
        second_action_data = json.loads(parsed_responses[1])
        second_action = second_action_data['arguments']['action']
        
        # Check pattern: first must be mouse_move or left_click, second must be left_click_drag
        if first_action not in ['mouse_move', 'left_click']:
            return []
        
        if second_action != 'left_click_drag':
            return []
        
        # Extract coordinates
        start_coordinate = first_action_data['arguments']['coordinate']
        end_coordinate = second_action_data['arguments']['coordinate']
        
        # Resize coordinates from model space to original image space
        start_coordinate = resize_coordinates(
            start_coordinate, size_info.resized_size, size_info.original_size
        )
        end_coordinate = resize_coordinates(
            end_coordinate, size_info.resized_size, size_info.original_size
        )
        
        # Return drag action
        return [{
            'action': 'drag',
            'coordinates': [start_coordinate, end_coordinate]
        }]
        
    except (json.JSONDecodeError, KeyError) as e:
        # Any parsing error returns empty
        return []


def draw_single_visualization(viz_task: Tuple[List[Dict[str, Any]], str, str]) -> None:
    """Draw actions on a single image - for multiprocessing"""
    actions, image_path, save_path = viz_task
    try:
        draw_actions_on_image(actions, image_path, save_path)
    except Exception as e:
        print(f"Error creating visualization for {save_path}: {e}")


def process_visualizations_parallel(viz_tasks: List[Tuple[List[Dict[str, Any]], str, str]], max_workers: int = 10) -> None:
    """Process multiple visualizations in parallel using multiprocessing"""
    if not viz_tasks:
        return
    
    print(f"Processing {len(viz_tasks)} visualizations with {max_workers} processes...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all visualization tasks
        futures = {executor.submit(draw_single_visualization, task): i for i, task in enumerate(viz_tasks)}
        
        # Wait for completion with progress
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exception that occurred
                completed += 1
            except Exception as e:
                task_index = futures[future]
                print(f"Visualization task {task_index} failed: {e}")
                completed += 1
    
    print(f"Completed {completed}/{len(viz_tasks)} visualizations")


def load_results_for_visualization(base_save_dir: str) -> List[Dict[str, Any]]:
    """Load saved results for visualization"""
    results = []
    if not os.path.exists(base_save_dir):
        print(f"Directory not found: {base_save_dir}")
        return results
    
    print(f"Scanning for JSON result files in: {base_save_dir}")
    json_files = []
    vis_files = []
    
    for filename in os.listdir(base_save_dir):
        file_path = os.path.join(base_save_dir, filename)
        
        # Check if it's a valid file (not directory)
        if not os.path.isfile(file_path):
            continue
            
        if filename.endswith('.json'):
            json_files.append(filename)
        elif filename.endswith('_vis.png'):
            vis_files.append(filename)
    
    print(f"Found {len(json_files)} JSON files and {len(vis_files)} existing visualization files")
    
    for filename in json_files:
        file_path = os.path.join(base_save_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
                # Check if result has processed_results and they are not empty
                if 'processed_results' in result and result['processed_results']:
                    # Check if corresponding image file exists
                    image_path = result.get('image_path', '')
                    if image_path and os.path.exists(image_path):
                        results.append(result)
                    else:
                        print(f"Warning: Image file not found for {filename}, image_path: {image_path}")
                        
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(results)} valid results with processed_results and existing images")
    return results


def run_visualization_task(base_save_dir: str, num_processes: int = 10):
    """Run visualization task on saved results"""
    print(f"Loading results from: {base_save_dir}")
    results = load_results_for_visualization(base_save_dir)
    
    if not results:
        print("No valid results with processed_results found!")
        print("Make sure you have run inference task first and that results contain drag actions.")
        return
    
    print(f"Found {len(results)} valid results with processed_results")
    
    # Prepare visualization tasks
    viz_tasks = []
    existing_viz_count = 0
    
    for result in results:
        if 'item_id' in result and 'processed_results' in result and 'image_path' in result:
            item_id = result['item_id']
            processed_results = result['processed_results']
            image_path = result['image_path']
            
            viz_save_path = os.path.join(base_save_dir, f"{item_id}_vis.png")
            
            # Only add if visualization doesn't exist
            if not os.path.exists(viz_save_path):
                viz_tasks.append((processed_results, image_path, viz_save_path))
            else:
                existing_viz_count += 1
    
    print(f"Found {existing_viz_count} existing visualizations")
    
    if not viz_tasks:
        print("All visualizations already exist!")
        return
    
    print(f"Creating {len(viz_tasks)} new visualizations using {min(num_processes, len(viz_tasks))} processes...")
    process_visualizations_parallel(viz_tasks, min(num_processes, len(viz_tasks)))


def draw_actions_on_image(actions: List[Dict[str, Any]], image_path: str, save_path: Optional[str] = None):
    """Draw action coordinates and annotations on image"""
    if len(actions) == 0:
        return
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Annotation parameters
    cross_size = 20
    circle_radius = 15
    cross_width = 2
    circle_width = 2
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Process each action
    for action_data in actions:
        action_name = action_data['action']
        
        # Get coordinates
        coordinates = []
        if action_name == 'drag':
            coordinates = action_data['coordinates']
        elif 'coordinate' in action_data:
            coordinates = [action_data['coordinate']]
        else:
            continue
        
        # Draw annotations at coordinate points
        for i, coord in enumerate(coordinates):
            x, y = coord
            
            # Draw red cross
            draw.line([x - cross_size, y - cross_size, x + cross_size, y + cross_size], 
                     fill='red', width=cross_width)
            draw.line([x - cross_size, y + cross_size, x + cross_size, y - cross_size], 
                     fill='red', width=cross_width)
            
            # Draw green circle
            draw.ellipse([x - circle_radius, y - circle_radius, 
                         x + circle_radius, y + circle_radius], 
                        outline='green', width=circle_width)
            
            # Add text label at first coordinate
            if i == 0:
                text_x = x + circle_radius + 25
                text_y = y - circle_radius - 25
                
                if font:
                    draw.text((text_x, text_y), action_name, fill='red', font=font)
                else:
                    draw.text((text_x, text_y), action_name, fill='red')
        
        # Draw connecting line for drag actions
        if action_name == 'drag' and len(coordinates) == 2:
            start_x, start_y = coordinates[0]
            end_x, end_y = coordinates[1]
            draw.line([start_x, start_y, end_x, end_y], fill='blue', width=3)
    
    # Display or save image
    if save_path:
        image.save(save_path)
        print(f"Visualization saved to: {save_path}")
    else:
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.title('Drag Action Visualization')
            plt.show()
        else:
            print("Matplotlib not available. Cannot display visualization.")


# vLLM service management
def start_vllm_service(model_path: str, port: int, model_name: str):
    """Start vLLM service"""
    command = [
        "vllm", "serve", model_path,
        "--served-model-name", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(torch.cuda.device_count()) if TORCH_AVAILABLE else "1",
        "--max-model-len", str(16384),
    ]
    return subprocess.Popen(command)


def wait_for_service(port: int, timeout: int = 600) -> bool:
    """Wait for service to start"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(("localhost", port))
            if result == 0:
                return True
        time.sleep(1)
    return False


def terminate_vllm_service(process):
    """Terminate vLLM service"""
    try:
        process.terminate()
        time.sleep(5)
        if process.poll() is None:
            process.kill()
            time.sleep(1)
    except Exception as e:
        print(f"Failed to terminate VLLM service: {e}")


# Model implementations
class DragModel_vLLM:
    """vLLM-based drag model"""
    
    def __init__(self, base_url: str, model_name: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available for vLLM")
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model_name = model_name

    def generate(self, image_path: str, instruction: str, temperature: float = 0.0, max_tokens: int = 1000) -> Tuple[List[Dict], str, SizeInfo]:
        """Generate drag response for single image"""
        batch_results = self.generate_batch([image_path], [instruction], temperature, max_tokens)
        return batch_results[0]

    def generate_batch(self, image_paths: List[str], instructions: List[str], temperature: float = 0.0, max_tokens: int = 1000) -> List[Tuple[List[Dict], str, SizeInfo]]:
        """Generate drag response for multiple images using multi-threading"""
        batch_size = len(image_paths)
        assert len(instructions) == batch_size, "Number of images and instructions must match"
        
        # Prepare requests
        requests = []
        size_infos = []
        
        for i, (image_path, instruction) in enumerate(zip(image_paths, instructions)):
            input_image = Image.open(image_path)
            resized_height, resized_width = qwen_smart_resize(
                input_image.height,
                input_image.width,
                max_pixels=2116800,
                min_pixels=12544,
            )
            
            size_info = SizeInfo(
                original_size=(input_image.width, input_image.height),
                resized_size=(resized_width, resized_height)
            )
            size_infos.append(size_info)
            
            messages = [
                {
                    "role": "system",
                    "content": FN_CALL_TEMPLATE.format(width=resized_width, height=resized_height)
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encode_image(input_image)}"}
                        },
                        {
                            "type": "text",
                            "text": f"{instruction}"
                        }
                    ]
                }
            ]
            
            request = {
                "index": i,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "size_info": size_info
            }
            requests.append(request)
        
        def process_single_request(request):
            """Process a single request with retry logic"""
            index = request["index"]
            messages = request["messages"]
            temperature = request["temperature"]
            max_tokens = request["max_tokens"]
            size_info = request["size_info"]
            response_text = ""
            
            for attempt in range(MAX_ATTEMPTS):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    response_text = response.choices[0].message.content
                    parsed_responses = extract_tool_calls(response_text)
                    
                    if parsed_responses:
                        processed_results = process_simple_drag_response(parsed_responses, size_info)
                        return index, (processed_results, response_text, size_info)
                    else:
                        raise ValueError(f"No tool calls found for request {index}, attempt {attempt + 1}")
                            
                except Exception as e:
                    print(f"Error during vLLM inference for request {index}: {str(e)}")
                    if attempt < MAX_ATTEMPTS - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
            
            # All attempts failed
            return index, ([], response_text, size_info)
        
        # Use ThreadPoolExecutor for concurrent processing
        results = [None] * batch_size
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all requests
            futures = {executor.submit(process_single_request, request): request["index"] 
                      for request in requests}
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as e:
                    index = futures[future]
                    print(f"Thread execution failed for request {index}: {str(e)}")
                    results[index] = ([], "", size_infos[index])
        
        return results


class DragModel_Claude:
    """Claude CUA (Computer Use Agent) model"""
    
    def __init__(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available")
            
        self.client = AnthropicBedrock(
            aws_region=os.getenv("AWS_REGION"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_key=os.getenv("AWS_SECRET_KEY"),
        )
        self.model_name = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        self.betas = ["computer-use-2025-01-24"]
        self.tools = [
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768,
                "display_number": 1,
            }
        ]

    def generate_batch(self, image_paths: List[str], instructions: List[str], enable_thinking: bool = False, without_hint: bool = False, max_concurrent: int = 10) -> List[Tuple[List[Dict], str, SizeInfo]]:
        """Generate drag response for multiple images using async processing"""
        return asyncio.run(self._generate_batch_async(image_paths, instructions, enable_thinking, without_hint, max_concurrent))

    async def _generate_batch_async(self, image_paths: List[str], instructions: List[str], enable_thinking: bool = False, without_hint: bool = False, max_concurrent: int = 10) -> List[Tuple[List[Dict], str, SizeInfo]]:
        """Async batch processing for Claude"""
        batch_size = len(image_paths)
        assert len(instructions) == batch_size, "Number of images and instructions must match"
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Use async context manager for proper resource cleanup
        async with AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_key=os.getenv("AWS_SECRET_KEY"),
        ) as async_client:
            
            async def process_single_claude_async(index: int, image_path: str, instruction: str):
                async with semaphore:
                    try:
                        # Add small delay to avoid hitting rate limits
                        await asyncio.sleep(0.1)
                        
                        # Process image
                        img = Image.open(image_path)
                        screenshot_width = img.width
                        screenshot_height = img.height
                        
                        resized_width, resized_height = self._scale_coordinates(
                            source="Computer", x=screenshot_width, y=screenshot_height,
                            width=screenshot_width, height=screenshot_height
                        )
                        
                        # Resize screenshot
                        resized_img = img.resize((resized_width, resized_height))
                        output_stream = io.BytesIO()
                        resized_img.save(output_stream, format=img.format or "PNG")
                        resized_screenshot = output_stream.getvalue()
                        
                        image_base64 = base64.b64encode(resized_screenshot).decode("utf-8")
                        
                        size_info = SizeInfo(
                            original_size=(screenshot_width, screenshot_height),
                            resized_size=(resized_width, resized_height)
                        )
                        
                        if without_hint:
                            prompt = f"You don't have to take screenshot as the input image is the screenshot. The instruction is: {instruction}"
                        else:
                            prompt = f"**IMPORTANT** Please perform drag action to select the target text span. Note that sentences usually end with a period, question mark, or exclamation mark. A sentence is not a line of text but rather a span of text properly delimited by punctuation. You cannot further take screenshot as this is the one to use.**IMPORTANT** The instruction is: {instruction}"

                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": image_base64
                                        },
                                    }
                                ]
                            }
                        ]
                        
                        args = {
                            "model": self.model_name,
                            "messages": messages,
                            "tools": self.tools,
                            "betas": self.betas,
                            "max_tokens": 1800,
                            "tool_choice": {"type": "auto"}
                        }
                        
                        if enable_thinking:
                            args["thinking"] = {"type": "enabled", "budget_tokens": 1024}
                        else:
                            args["thinking"] = {"type": "disabled"}
                        
                        # Use async Claude client
                        response = await async_client.beta.messages.create(**args)
                        response_json = json.loads(response.model_dump_json())
                        
                        # Extract drag coordinates from Claude response
                        actions = []
                        for item in response_json["content"]:
                            if item["type"] == "tool_use":
                                action_type = item["input"]["action"]
                                if action_type == "left_click_drag":
                                    start_x, start_y = item["input"]["start_coordinate"]
                                    end_x, end_y = item["input"]["coordinate"]
                                    
                                    # Scale coordinates back to original size
                                    start_x, start_y = self._scale_coordinates(
                                        "API", start_x, start_y, 
                                        size_info.original_size[0], size_info.original_size[1]
                                    )
                                    end_x, end_y = self._scale_coordinates(
                                        "API", end_x, end_y,
                                        size_info.original_size[0], size_info.original_size[1]
                                    )
                                    
                                    actions.append({
                                        'action': 'drag',
                                        'coordinates': [(start_x, start_y), (end_x, end_y)]
                                    })
                        
                        return index, (actions, str(response_json), size_info)
                        
                    except Exception as e:
                        print(f"Error in Claude async processing for image {index}: {e}")
                        img = Image.open(image_path)
                        size_info = SizeInfo(
                            original_size=(img.width, img.height),
                            resized_size=(img.width, img.height)
                        )
                        return index, ([], "", size_info)
            
            # Create tasks for all images
            tasks = [
                process_single_claude_async(i, image_path, instruction)
                for i, (image_path, instruction) in enumerate(zip(image_paths, instructions))
            ]
            
            # Execute with progress bar
            results = [None] * batch_size
            
            pbar = async_tqdm(total=len(tasks), desc="Claude Processing", unit="item")
            
            for coro in asyncio.as_completed(tasks):
                index, result = await coro
                results[index] = result
                pbar.update(1)
            
            pbar.close()
            return results

    def _scale_coordinates(self, source: str, x: int, y: int, width: int, height: int) -> Tuple[int, int]:
        """Scale coordinates for Claude CUA"""
        MAX_SCALING_TARGETS = {
            "XGA": {"width": 1024, "height": 768},  # 4:3
            "WXGA": {"width": 1280, "height": 800},  # 16:10
            "FWXGA": {"width": 1366, "height": 768},  # ~16:9
        }
        
        ratio = width / height
        target_dimension = min(
            MAX_SCALING_TARGETS.values(), 
            key=lambda d: abs(abs(d["width"] / d["height"]) - ratio)
        )
        
        x_scaling_factor = target_dimension["width"] / width
        y_scaling_factor = target_dimension["height"] / height
        
        if source == "API":
            if x > width or y > height:
                print(f"Warning: Coordinates {x}, {y} are out of bounds")
                x = min(x, width)
                y = min(y, height)
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        elif source == "Computer":
            return round(x * x_scaling_factor), round(y * y_scaling_factor)
        else:
            raise ValueError(f"Invalid source {source}")

    def generate(self, image_path: str, instruction: str, enable_thinking: bool = False, without_hint: bool = False) -> Tuple[List[Dict], str, SizeInfo]:
        """Generate drag response for single image"""
        batch_results = self.generate_batch([image_path], [instruction], enable_thinking, without_hint)
        return batch_results[0]

    def _generate_single(self, image_path: str, instruction: str, enable_thinking: bool = False, without_hint: bool = False) -> Tuple[List[Dict], str, SizeInfo]:
        """Generate drag response"""
        img = Image.open(image_path)
        screenshot_width = img.width
        screenshot_height = img.height
        
        resized_width, resized_height = self._scale_coordinates(
            source="Computer", x=screenshot_width, y=screenshot_height,
            width=screenshot_width, height=screenshot_height
        )
        
        # Resize screenshot
        resized_img = img.resize((resized_width, resized_height))
        output_stream = io.BytesIO()
        resized_img.save(output_stream, format=img.format or "PNG")
        resized_screenshot = output_stream.getvalue()
        
        image_base64 = base64.b64encode(resized_screenshot).decode("utf-8")
        
        size_info = SizeInfo(
            original_size=(screenshot_width, screenshot_height),
            resized_size=(resized_width, resized_height)
        )
        
        if without_hint:
            prompt = f"The instruction is: {instruction}"
        else:
            prompt = f"**IMPORTANT** Please perform drag action to select the target text span. Note that sentences usually end with a period, question mark, or exclamation mark. A sentence is not a line of text but rather a span of text properly delimited by punctuation. You cannot further take screenshot as this is the one to use.**IMPORTANT** The instruction is: {instruction}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        },
                    }
                ]
            }
        ]
        
        args = {
            "model": self.model_name,
            "messages": messages,
            "tools": self.tools,
            "betas": self.betas,
            "max_tokens": 1800,
            "tool_choice": {"type": "auto"}
        }
        
        if enable_thinking:
            args["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        else:
            args["thinking"] = {"type": "disabled"}
        
        try:
            response = self.client.beta.messages.create(**args)
            response_json = json.loads(response.model_dump_json())
            
            # Extract drag coordinates from Claude response
            actions = []
            for item in response_json["content"]:
                if item["type"] == "tool_use":
                    action_type = item["input"]["action"]
                    if action_type == "left_click_drag":
                        start_x, start_y = item["input"]["start_coordinate"]
                        end_x, end_y = item["input"]["coordinate"]
                        
                        # Scale coordinates back to original size
                        start_x, start_y = self._scale_coordinates(
                            "API", start_x, start_y, 
                            size_info.original_size[0], size_info.original_size[1]
                        )
                        end_x, end_y = self._scale_coordinates(
                            "API", end_x, end_y,
                            size_info.original_size[0], size_info.original_size[1]
                        )
                        
                        actions.append({
                            'action': 'drag',
                            'coordinates': [(start_x, start_y), (end_x, end_y)]
                        })
            
            return actions, str(response_json), size_info
            
        except Exception as e:
            print(f"Error during Claude inference: {str(e)}")
            return [], "", size_info


class DragModel_Operator:
    """OpenAI Operator (Computer Use Preview) model"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available")
            
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = "computer-use-preview"

    def generate(self, image_path: str, instruction: str, enable_thinking: bool = False, without_hint: bool = False, max_attempts: int = 5) -> Tuple[List[Dict], str, SizeInfo]:
        """Generate drag response for single image"""
        batch_results = self.generate_batch([image_path], [instruction], enable_thinking, without_hint)
        return batch_results[0]

    def _generate_single(self, image_path: str, instruction: str, enable_thinking: bool = False, without_hint: bool = False, max_attempts: int = 5) -> Tuple[List[Dict], str, SizeInfo]:
        """Generate drag response for single image (internal method)"""
        input_image = Image.open(image_path)
        img_width, img_height = input_image.size
        img_base64 = encode_image(input_image)
        
        size_info = SizeInfo(
            original_size=(img_width, img_height),
            resized_size=(img_width, img_height)
        )
        
        if without_hint:
            prompt = f"The instruction is: {instruction}"
        else:
            prompt = f"**IMPORTANT** Please perform drag action to select the target text span. Note that sentences usually end with a period, question mark, or exclamation mark. A sentence is not a line of text but rather a span of text properly delimited by punctuation. **IMPORTANT** The instruction is: {instruction}"
        
        for attempt in range(max_attempts):
            try:
                if enable_thinking:
                    response = self.client.responses.create(
                        model=self.model_name,
                        tools=[
                            {
                                "type": "computer_use_preview",
                                "display_width": img_width,
                                "display_height": img_height,
                                "environment": "linux",
                            }
                        ],
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/png;base64,{img_base64}",
                                    },
                                    {"type": "input_text", "text": prompt},
                                ],
                            }
                        ],
                        reasoning={"generate_summary": "concise"},
                        truncation="auto",
                    )
                else:
                    response = self.client.responses.create(
                        model=self.model_name,
                        tools=[
                            {
                                "type": "computer_use_preview",
                                "display_width": img_width,
                                "display_height": img_height,
                                "environment": "linux",
                            }
                        ],
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/png;base64,{img_base64}",
                                    },
                                    {"type": "input_text", "text": prompt},
                                ],
                            }
                        ],
                        truncation="auto",
                    )
                
                response_json = json.loads(response.model_dump_json())
                response_output = response_json["output"]
                
                # Extract drag coordinates from Operator response
                actions = []
                for obj in response_output:
                    if obj["type"] == "computer_call" and obj["action"]["type"] == "drag":
                        path = obj["action"]["path"]
                        if len(path) >= 2:
                            start_x = path[0]["x"]
                            start_y = path[0]["y"]
                            end_x = path[-1]["x"]
                            end_y = path[-1]["y"]
                            
                            actions.append({
                                'action': 'drag',
                                'coordinates': [(start_x, start_y), (end_x, end_y)]
                            })
                
                return actions, str(response_json), size_info
                
            except Exception as e:
                print(f"Error during Operator inference: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                    
        return [], "", size_info

    def generate_batch(self, image_paths: List[str], instructions: List[str], enable_thinking: bool = False, without_hint: bool = False, max_concurrent: int = 10) -> List[Tuple[List[Dict], str, SizeInfo]]:
        """Generate drag response for multiple images using async processing"""
        return asyncio.run(self._generate_batch_async(image_paths, instructions, enable_thinking, without_hint, max_concurrent))

    async def _generate_batch_async(self, image_paths: List[str], instructions: List[str], enable_thinking: bool = False, without_hint: bool = False, max_concurrent: int = 10) -> List[Tuple[List[Dict], str, SizeInfo]]:
        """Async batch processing for Operator"""
        batch_size = len(image_paths)
        assert len(instructions) == batch_size, "Number of images and instructions must match"
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Use async context manager for proper resource cleanup
        async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as async_client:
            
            async def process_single_operator_async(index: int, image_path: str, instruction: str):
                async with semaphore:
                    try:
                        # Add small delay to avoid hitting rate limits
                        await asyncio.sleep(0.1)
                        
                        # Process image
                        input_image = Image.open(image_path)
                        img_width, img_height = input_image.size
                        img_base64 = encode_image(input_image)
                        
                        size_info = SizeInfo(
                            original_size=(img_width, img_height),
                            resized_size=(img_width, img_height)
                        )
                        
                        if without_hint:
                            prompt = f"The instruction is: {instruction}"
                        else:
                            prompt = f"**IMPORTANT** Please perform drag action to select the target text span. Note that sentences usually end with a period, question mark, or exclamation mark. A sentence is not a line of text but rather a span of text properly delimited by punctuation. **IMPORTANT** The instruction is: {instruction}"
                        
                        # Use async OpenAI client
                        if enable_thinking:
                            response = await async_client.responses.create(
                                model=self.model_name,
                                tools=[
                                    {
                                        "type": "computer_use_preview",
                                        "display_width": img_width,
                                        "display_height": img_height,
                                        "environment": "linux",
                                    }
                                ],
                                input=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_image",
                                                "image_url": f"data:image/png;base64,{img_base64}",
                                            },
                                            {"type": "input_text", "text": prompt},
                                        ],
                                    }
                                ],
                                reasoning={"generate_summary": "concise"},
                                truncation="auto",
                            )
                        else:
                            response = await async_client.responses.create(
                                model=self.model_name,
                                tools=[
                                    {
                                        "type": "computer_use_preview",
                                        "display_width": img_width,
                                        "display_height": img_height,
                                        "environment": "linux",
                                    }
                                ],
                                input=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_image",
                                                "image_url": f"data:image/png;base64,{img_base64}",
                                            },
                                            {"type": "input_text", "text": prompt},
                                        ],
                                    }
                                ],
                                truncation="auto",
                            )
                        
                        response_json = json.loads(response.model_dump_json())
                        response_output = response_json["output"]
                        
                        # Extract drag coordinates from Operator response
                        actions = []
                        for obj in response_output:
                            if obj["type"] == "computer_call" and obj["action"]["type"] == "drag":
                                path = obj["action"]["path"]
                                if len(path) >= 2:
                                    start_x = path[0]["x"]
                                    start_y = path[0]["y"]
                                    end_x = path[-1]["x"]
                                    end_y = path[-1]["y"]
                                    
                                    actions.append({
                                        'action': 'drag',
                                        'coordinates': [(start_x, start_y), (end_x, end_y)]
                                    })
                        
                        return index, (actions, str(response_json), size_info)
                        
                    except Exception as e:
                        print(f"Error in Operator async processing for image {index}: {e}")
                        img = Image.open(image_path)
                        size_info = SizeInfo(
                            original_size=(img.width, img.height),
                            resized_size=(img.width, img.height)
                        )
                        return index, ([], "", size_info)
            
            # Create tasks for all images
            tasks = [
                process_single_operator_async(i, image_path, instruction)
                for i, (image_path, instruction) in enumerate(zip(image_paths, instructions))
            ]
            
            # Execute with progress bar
            results = [None] * batch_size
            
            pbar = async_tqdm(total=len(tasks), desc="Operator Processing", unit="item")
            
            for coro in asyncio.as_completed(tasks):
                index, result = await coro
                results[index] = result
                pbar.update(1)
            
            pbar.close()
            return results




class DragModel_UITAR:
    """UITAR model with official coordinate processing"""
    
    def __init__(self, base_url: str, model_name: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available for UITAR")
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model_name = model_name

    def generate(self, image_path: str, instruction: str, temperature: float = 0.0, max_tokens: int = 1000) -> Tuple[List[Dict], str, SizeInfo]:
        """Generate drag response using UITAR format with official coordinate processing"""
        input_image = Image.open(image_path)
        original_width, original_height = input_image.size
        
        # Use UITAR-specific smart resize
        resized_height, resized_width = uitar_smart_resize(original_height, original_width)
        
        size_info = SizeInfo(
            original_size=(original_width, original_height),
            resized_size=(resized_width, resized_height)
        )
        
        # Use UITAR-specific prompt format
        formatted_instruction = UITARS_USR_PROMPT_THOUGHT.format(instruction=instruction)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(input_image)}"}
                    },
                    {
                        "type": "text",
                        "text": formatted_instruction
                    }
                ]
            }
        ]
        
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                response_text = response.choices[0].message.content
                
                # Parse UITAR response format
                actions = []
                
                # Parse drag action with box format: drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
                drag_box_pattern = r"drag\(start_box='\((\d+),\s*(\d+)\)',\s*end_box='\((\d+),\s*(\d+)\)'\)"
                drag_box_match = re.search(drag_box_pattern, response_text)
                
                if drag_box_match:
                    start_x, start_y, end_x, end_y = map(int, drag_box_match.groups())
                    
                    # Convert model coordinates to original image coordinates using official method
                    start_x_orig = int(start_x / resized_width * original_width)
                    start_y_orig = int(start_y / resized_height * original_height)
                    end_x_orig = int(end_x / resized_width * original_width)
                    end_y_orig = int(end_y / resized_height * original_height)
                    
                    actions.append({
                        'action': 'drag',
                        'coordinates': [(start_x_orig, start_y_orig), (end_x_orig, end_y_orig)]
                    })
                    return actions, response_text, size_info
                
            except Exception as e:
                print(f"Error during UITAR inference: {str(e)}")
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                    
        return [], response_text, size_info



def evaluate_single_image(model, image_path: str, instruction: str, save_dir: Optional[str] = None, model_save_name: Optional[str] = None, enable_thinking: bool = False, without_hint: bool = False, item_id: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """Evaluate drag task on a single image"""
    if verbose:
        print(f"Evaluating: {instruction}")
        print(f"Image: {image_path}")
    
    # Generate based on model type
    if hasattr(model, 'generate'):
        if isinstance(model, DragModel_Claude):
            processed_results, response_text, size_info = model.generate(image_path, instruction, enable_thinking=enable_thinking, without_hint=without_hint)
        elif isinstance(model, DragModel_Operator):
            processed_results, response_text, size_info = model.generate(image_path, instruction, enable_thinking=enable_thinking, without_hint=without_hint)
        else:
            processed_results, response_text, size_info = model.generate(image_path, instruction)
    else:
        raise ValueError(f"Model {type(model)} does not have generate method")
    
    # Print results
    if processed_results:
        if verbose:
            print("Drag action detected!")
            for result in processed_results:
                if result['action'] == 'drag':
                    start_coord = result['coordinates'][0]
                    end_coord = result['coordinates'][1]
                    print(f"Drag from {start_coord} to {end_coord}")
        
        # Note: Visualization is handled separately in viz task
    else:
        if verbose:
            print("No drag action detected or parsing failed")
            print(f"Raw response preview: {response_text[:500]}...")
    
    result = {
        'item_id': item_id,
        'instruction': instruction,
        'image_path': image_path,
        'processed_results': processed_results,
        'response_text': response_text,
        'size_info': size_info.__dict__
    }
    
    # Add basic debug info for all models
    result['debug_info'] = {
        'raw_response_text': response_text,
        'parse_success': len(processed_results) > 0,
        'model_type': type(model).__name__
    }
    
    return result


def load_benchmark(benchmark_path: str) -> List[Dict[str, Any]]:
    """Load benchmark data from JSON file"""
    print(f"Loading benchmark from: {benchmark_path}")
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test cases")
    return data


def evaluate_benchmark(model, benchmark_data: List[Dict[str, Any]], save_dir: Optional[str] = None, 
                      model_save_name: Optional[str] = None, enable_thinking: bool = False, 
                      without_hint: bool = False, batch_size: int = 1, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """Evaluate model on entire benchmark with smart skip based on existing files"""
    
    total_items = len(benchmark_data)
    print(f"Evaluating {total_items} items from benchmark")
    
    # Setup results storage
    results = []
    base_save_dir = None
    
    if save_dir:
        # Create model-specific subdirectory if model_save_name is provided
        if model_save_name:
            # Add backend and thinking info to folder name
            folder_parts = [model_save_name]
            
            # Determine backend from model type
            if isinstance(model, DragModel_vLLM):
                folder_parts.append("vllm")
            elif isinstance(model, DragModel_Claude):
                folder_parts.append("claude")
            elif isinstance(model, DragModel_Operator):
                folder_parts.append("operator")
            elif isinstance(model, DragModel_UITAR):
                folder_parts.append("uitar")
            else:
                folder_parts.append("unknown")
            
            # Add thinking info
            if enable_thinking:
                folder_parts.append("thinking_true")
            else:
                folder_parts.append("thinking_false")
            
            # Add without_hint info for claude and operator models
            if without_hint and isinstance(model, (DragModel_Claude, DragModel_Operator)):
                folder_parts.append("without_hint")
            
            folder_name = "_".join(folder_parts)
            model_dir = os.path.join(save_dir, folder_name)
            os.makedirs(model_dir, exist_ok=True)
            base_save_dir = model_dir
        else:
            os.makedirs(save_dir, exist_ok=True)
            base_save_dir = save_dir
    
    # Function to check if individual result exists
    def get_individual_result_path(item_id: str) -> Optional[str]:
        if not base_save_dir:
            return None
        
        # Simple filename: {item_id}.json
        individual_result_path = os.path.join(base_save_dir, f"{item_id}.json")
        return individual_result_path if os.path.exists(individual_result_path) else None
    
    # Load existing results from individual files if available
    existing_results = {}
    if base_save_dir and os.path.exists(base_save_dir):
        print(f"Checking for existing individual result files in: {base_save_dir}")
        try:
            # Load all existing .json files
            for filename in os.listdir(base_save_dir):
                if filename.endswith('.json') and not filename.endswith('_vis.png'):
                    item_id = filename[:-5]  # Remove .json extension
                    file_path = os.path.join(base_save_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                        existing_results[item_id] = result
                    except Exception as e:
                        print(f"Warning: Could not load existing result {filename}: {e}")
            print(f"Found {len(existing_results)} existing individual result files")
        except Exception as e:
            print(f"Warning: Could not scan for existing results: {e}")
            existing_results = {}
    
    # Progress tracking
    processed_count = 0
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Check if the model supports batch processing
    supports_batch = hasattr(model, 'generate_batch') and isinstance(
        model,
        (
            DragModel_vLLM,
            DragModel_Operator,
            DragModel_Claude,
        ),
    )
    
    if supports_batch and batch_size > 1:
        print(f"Using batch processing with batch_size={batch_size}")
    else:
        print("Using single item processing")
        batch_size = 1
    
    # Process benchmark items
    with tqdm(total=total_items, desc="Evaluating benchmark") as pbar:
        i = 0
        while i < total_items:
            # Collect batch items (skip already processed ones)
            batch_items = []
            batch_item_ids = []
            batch_image_paths = []
            batch_instructions = []
            
            while len(batch_items) < batch_size and i < total_items:
                item = benchmark_data[i]
                # Use item_id from benchmark data if available, otherwise use index
                item_id = str(item.get('item_id', i))
                i += 1
                
                # Check if result already exists
                if item_id in existing_results:
                    results.append(existing_results[item_id])
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                # Check if individual result file exists
                individual_result_path = get_individual_result_path(item_id)
                if individual_result_path:
                    try:
                        with open(individual_result_path, 'r', encoding='utf-8') as f:
                            individual_result = json.load(f)
                        results.append(individual_result)
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    except Exception as e:
                        print(f"Warning: Could not load individual result {individual_result_path}: {e}")
                
                # Check if image exists
                image_path = item['annotation_path']
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    error_result = {
                        'item_id': item_id,
                        'instruction': item['expression'],
                        'image_path': image_path,
                        'error': 'Image file not found',
                        'processed_results': [],
                        'response_text': '',
                        'size_info': {},
                        'debug_info': {
                            'raw_response_text': '',
                            'parse_success': False,
                            'error': 'Image file not found',
                            'model_type': type(model).__name__
                        }
                    }
                    results.append(error_result)
                    
                    # Save individual error result file immediately
                    if base_save_dir:
                        individual_result_path = os.path.join(base_save_dir, f"{item_id}.json")
                        try:
                            with open(individual_result_path, 'w', encoding='utf-8') as f:
                                json.dump(error_result, f, indent=2, ensure_ascii=False)
                        except Exception as save_e:
                            print(f"Warning: Could not save error result for item {item_id}: {save_e}")
                    
                    error_count += 1
                    pbar.update(1)
                    continue
                
                # Add to batch
                batch_items.append(item)
                batch_item_ids.append(item_id)
                batch_image_paths.append(image_path)
                batch_instructions.append(item['expression'])
            
            # Process batch
            if batch_items:
                try:
                    if supports_batch and len(batch_items) > 1:
                        if isinstance(model, DragModel_vLLM):
                            batch_results = model.generate_batch(
                                batch_image_paths,
                                batch_instructions,
                            )
                        elif isinstance(model, DragModel_Operator):
                            batch_results = model.generate_batch(
                                batch_image_paths,
                                batch_instructions,
                                enable_thinking=enable_thinking,
                                without_hint=without_hint,
                                max_concurrent=max_concurrent,
                            )
                        elif isinstance(model, DragModel_Claude):
                            batch_results = model.generate_batch(
                                batch_image_paths,
                                batch_instructions,
                                enable_thinking=enable_thinking,
                                without_hint=without_hint,
                                max_concurrent=max_concurrent,
                            )
                        else:
                            batch_results = []
                    else:
                        batch_results = []
                        for j, _ in enumerate(batch_items):
                            if isinstance(model, DragModel_Claude):
                                single_result = model.generate(
                                    batch_image_paths[j],
                                    batch_instructions[j],
                                    enable_thinking=enable_thinking,
                                    without_hint=without_hint,
                                )
                            elif isinstance(model, DragModel_Operator):
                                single_result = model.generate(
                                    batch_image_paths[j],
                                    batch_instructions[j],
                                    enable_thinking=enable_thinking,
                                    without_hint=without_hint,
                                )
                            else:
                                single_result = model.generate(
                                    batch_image_paths[j],
                                    batch_instructions[j],
                                )
                            batch_results.append(single_result)

                    for j, (item_id, item) in enumerate(zip(batch_item_ids, batch_items)):
                        processed_results, response_text, size_info = batch_results[j]
                        debug_info = {
                            'raw_response_text': response_text,
                            'parse_success': len(processed_results) > 0,
                            'model_type': type(model).__name__,
                        }

                        # Create result dict
                        result = {
                            'item_id': item_id,
                            'instruction': item['expression'],
                            'image_path': item['annotation_path'],
                            'processed_results': processed_results,
                            'response_text': response_text,
                            'size_info': size_info.__dict__ if hasattr(size_info, '__dict__') else size_info,
                        }

                        # Attach debug info
                        result['debug_info'] = debug_info

                        # Add additional benchmark info
                        result.update({
                            'ids_of_the_bboxes': item.get('ids_of_the_bboxes', []),
                            'combined_bboxes': item.get('combined_bboxes', []),
                            'grounded_path': item.get('grounded_path', ''),
                            'parsed_path': item.get('parsed_path', ''),
                            'filter_path': item.get('filter_path', ''),
                            'target_text_span': item.get('target_text_span', ''),
                        })

                        results.append(result)

                        # Save individual result file immediately
                        if base_save_dir:
                            individual_result_path = os.path.join(base_save_dir, f"{item_id}.json")
                            try:
                                with open(individual_result_path, 'w', encoding='utf-8') as f:
                                    json.dump(result, f, indent=2, ensure_ascii=False)
                            except Exception as e:
                                print(f"Warning: Could not save individual result for item {item_id}: {e}")

                        if processed_results:
                            success_count += 1

                        processed_count += 1

                        pbar.update(1)


                    
                    # Update progress
                    pbar.set_postfix({
                        'Processed': processed_count,
                        'Success': success_count,
                        'Errors': error_count,
                        'Skipped': skipped_count,
                        'Success Rate': f"{success_count/(processed_count or 1)*100:.1f}%"
                    })
                    
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    # Add error results for all items in batch
                    for item_id, item in zip(batch_item_ids, batch_items):
                        error_result = {
                            'item_id': item_id,
                            'instruction': item.get('expression', ''),
                            'image_path': item.get('annotation_path', ''),
                            'error': str(e),
                            'processed_results': [],
                            'response_text': '',
                            'size_info': {},
                            'debug_info': {
                                'raw_response_text': '',
                                'parse_success': False,
                                'error': str(e),
                                'model_type': type(model).__name__
                            }
                        }
                        results.append(error_result)
                        
                        # Save individual error result file immediately
                        if base_save_dir:
                            individual_result_path = os.path.join(base_save_dir, f"{item_id}.json")
                            try:
                                with open(individual_result_path, 'w', encoding='utf-8') as f:
                                    json.dump(error_result, f, indent=2, ensure_ascii=False)
                            except Exception as save_e:
                                print(f"Warning: Could not save error result for item {item_id}: {save_e}")
                        
                        error_count += 1
                        pbar.update(1)
    
    # Individual results are already saved to separate files
    if save_dir:
        print(f"Individual results saved to: {base_save_dir}")
        print(f"Each result saved as {{item_id}}.json")
    
    # Print summary
    print(f"\nBenchmark Evaluation Summary:")
    print(f"Total items in benchmark: {total_items}")
    print(f"Items skipped (already processed): {skipped_count}")
    print(f"Items newly processed: {processed_count}")
    print(f"Successful predictions: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Overall success rate: {success_count/(processed_count or 1)*100:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive drag model evaluation")
    
    # Task selection
    parser.add_argument("--task", 
                       choices=["inference", "viz"], 
                       required=True,
                       help="Task to perform: inference or visualization")
    
    # Model selection (for inference task)
    parser.add_argument("--backend", 
                       choices=["vllm", "claude", "operator", "uitar"],
                       help="Model backend to use (required for inference task)")
    
    # Input mode selection
    parser.add_argument("--benchmark", help="Path to benchmark.json file for batch evaluation")
    parser.add_argument("--image", help="Path to image (for single image evaluation)", default="/users/PAA0201/lzy37ld/LLaMA-Factory/lzy/qwen25vl.png")
    parser.add_argument("--instruction", help="Drag instruction (for single image evaluation)", default="Drag to select the first sentence of the paragraph.")
    
    # === Visualization Task Args ===
    parser.add_argument("--viz_processes", type=int, default=10, help="Number of processes for visualization (default: 10)")
    
    # === Inference Task Args ===
    # Batch processing
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument("--max_concurrent", type=int, default=10, help="Max concurrent API calls for Operator/Claude (default: 10)")
    
    # Model configuration
    parser.add_argument("--model_path", help="Path to model (required to auto-start local vLLM server)")
    parser.add_argument("--model_save_name", help="Model identifier for result directory naming")
    
    # vLLM/UITAR specific
    parser.add_argument("--base_url", default="http://localhost:1053/v1", help="vLLM base URL")
    parser.add_argument("--model_name", default="qwen25vl", help="Model name for vLLM")
    parser.add_argument("--port", type=int, default=1053, help="Port for vLLM service")
    parser.add_argument("--use_external_vllm", action="store_true", help="Use external vLLM service")
    
    # Generation parameters
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode for Claude and Operator models")
    parser.add_argument("--without_hint", action="store_true", help="Remove hint about sentences/punctuation from prompt for Claude and Operator models")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    
    # Output configuration
    parser.add_argument("--save_dir", help="Directory to save inference results")
    parser.add_argument("--output_file", help="Save single image result to JSON file")
    
    args = parser.parse_args()
    
    # Handle visualization task
    if args.task == "viz":
        # Auto-generate viz directory from inference parameters
        if not args.save_dir:
            raise SystemExit("--save_dir is required for visualization task")
        if not args.model_save_name:
            raise SystemExit("--model_save_name is required for visualization task")
        if not args.backend:
            raise SystemExit("--backend is required for visualization task to determine folder structure")
        
        # Generate the same directory structure as inference task
        folder_parts = [args.model_save_name]
        folder_parts.append(args.backend)
        
        if args.enable_thinking:
            folder_parts.append("thinking_true")
        else:
            folder_parts.append("thinking_false")
        
        # Add without_hint for claude and operator
        if args.without_hint and args.backend in ["claude", "operator"]:
            folder_parts.append("without_hint")
        
        folder_name = "_".join(folder_parts)
        viz_dir = os.path.join(args.save_dir, folder_name)
        
        print(f"Starting visualization task...")
        print(f"Auto-detected inference results directory: {viz_dir}")
        run_visualization_task(viz_dir, args.viz_processes)
        print("Visualization task completed!")
        return
    
    # Handle inference task
    if args.task == "inference":
        if not args.backend:
            raise SystemExit("--backend is required for inference task")
    
    # Initialize model based on backend
    model = None
    process = None
    
    try:
        if args.backend == "vllm":
            if not args.use_external_vllm:
                if not args.model_path:
                    raise SystemExit("--model_path required to auto-start vLLM")
                print(f"Starting local vLLM on port {args.port}...")
                process = start_vllm_service(args.model_path, args.port, args.model_name)
                if not wait_for_service(args.port):
                    raise SystemExit(f"Failed to start vLLM on port {args.port}")
                args.base_url = f"http://localhost:{args.port}/v1"
            
            model = DragModel_vLLM(args.base_url, args.model_name)
            
        elif args.backend == "claude":
            model = DragModel_Claude()
            
        elif args.backend == "operator":
            model = DragModel_Operator()
            
        elif args.backend == "uitar":
            model = DragModel_UITAR(args.base_url, args.model_name)
        
        # Determine evaluation mode
        if args.benchmark:
            # Batch evaluation mode
            print("Running batch evaluation on benchmark...")
            
            # Load benchmark data
            benchmark_data = load_benchmark(args.benchmark)
            
            # Run batch evaluation
            results = evaluate_benchmark(
                model=model,
                benchmark_data=benchmark_data,
                save_dir=args.save_dir,
                model_save_name=args.model_save_name,
                enable_thinking=args.enable_thinking,
                without_hint=args.without_hint,
                batch_size=args.batch_size,
                max_concurrent=args.max_concurrent
            )
            
            print("Batch evaluation completed!")
            
        else:
            # Single image evaluation mode
            print("Running single image evaluation...")
            
            result = evaluate_single_image(
                model, 
                args.image, 
                args.instruction,
                save_dir=args.save_dir,
                model_save_name=args.model_save_name,
                enable_thinking=args.enable_thinking,
                without_hint=args.without_hint
            )
            
            # Note: Visualization is handled separately in viz task
            
            # Save results based on save_dir or output_file
            if args.save_dir:
                # Create model-specific subdirectory if model_save_name is provided
                if args.model_save_name:
                    model_dir = os.path.join(args.save_dir, args.model_save_name)
                    os.makedirs(model_dir, exist_ok=True)
                    base_save_dir = model_dir
                else:
                    os.makedirs(args.save_dir, exist_ok=True)
                    base_save_dir = args.save_dir
                
                # Extract image name without extension (remove last .png)
                image_name = os.path.basename(args.image)
                if image_name.lower().endswith('.png'):
                    image_name = image_name[:-4]  # Remove last .png
                
                # Add thinking mode and without_hint to filename
                filename_parts = [image_name]
                if args.enable_thinking:
                    filename_parts.append("thinking_true")
                else:
                    filename_parts.append("thinking_false")
                
                # Add without_hint for claude and operator models
                if args.without_hint and args.backend in ["claude", "operator"]:
                    filename_parts.append("without_hint")
                
                filename_parts.append("result")
                
                output_path = os.path.join(base_save_dir, f"{'_'.join(filename_parts)}.json")
                
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                print(f"Results saved to {output_path}")
            elif args.output_file:
                os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
                with open(args.output_file, 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                print(f"Results saved to {args.output_file}")
            
            print("Single image evaluation completed!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        
    finally:
        if process:
            print("Terminating local vLLM service...")
            terminate_vllm_service(process)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    main()
