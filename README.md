# Python Kandinsky API

#### Async Multi-Key AI Image Generator Kandinsky API 

*   ⚡ Asynchronous requests
*   🔑 Supports multiple API keys
*   📊 Progress tracking
*   🖼️ Automatic image saving with unique names
*   ⚙️ Configurable image size and count

## Quick Start

1. Install requirements
    
    ```bash
    pip install aiohttp tqdm
    ```
    
2. Create keys.txt with your API keys (format: token:key)

    ```bash
    echo "YOUR_TOKEN:YOUR_KEY" > keys.txt
    ```
    
3. Run the generator

    ```bash
    python main.py
    ```

### Configuration


Edit these variables in `main.py`:

*   `prompt` - Your generation prompt
*   `total_images_per_key` - Images to generate per API key
*   `output_dir` - Output directory (default: "output")

### Keywords

AI, image-generation, fusionbrain, kandinsky, python, async, api, bulk-processing, content-creation, dataset-generator, neural-networks, automation
