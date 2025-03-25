import json
import time
import os
import base64
import asyncio
import aiohttp
import uuid
from aiohttp import FormData
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

class ProgressTracker:
    def __init__(self, total_images, total_keys):
        self.total = total_images * total_keys
        self.completed = 0
        self.lock = asyncio.Lock()
        self.key_stats = defaultdict(int)
        self.start_time = time.time()
        
        if HAS_TQDM:
            self.pbar = tqdm(total=self.total, desc="Генерация изображений", unit="img")
        else:
            print(f"Всего изображений для генерации: {self.total}")
    
    async def update(self, key_prefix, current, total):
        async with self.lock:
            self.completed += 1
            self.key_stats[key_prefix] += 1
            
            if HAS_TQDM:
                self.pbar.update(1)
                self.pbar.set_postfix_str(f"Ключ {key_prefix}: {self.key_stats[key_prefix]}/{total}")
            else:
                elapsed = time.time() - self.start_time
                img_per_sec = self.completed / elapsed if elapsed > 0 else 0
                print(f"\rПрогресс: {self.completed}/{self.total} ({self.completed/self.total:.1%}) | "
                      f"{img_per_sec:.2f} img/sec | "
                      f"Ключ {key_prefix}: {self.key_stats[key_prefix]}/{total}", end="")
    
    def close(self):
        if HAS_TQDM:
            self.pbar.close()
        else:
            print()

class Text2ImageAPI:
    def __init__(self, url, fusion_brain_token, fusion_brain_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {fusion_brain_token}',
            'X-Secret': f'Secret {fusion_brain_key}',
        }
        self.session = aiohttp.ClientSession()
        self.request_semaphore = asyncio.Semaphore(1)
        self.last_request_time = 0

    async def _throttled_request(self, method, url, **kwargs):
        async with self.request_semaphore:
            elapsed = time.time() - self.last_request_time
            if elapsed < 0.05:
                await asyncio.sleep(0.1 - elapsed)
            
            self.last_request_time = time.time()
            async with method(url, headers=self.AUTH_HEADERS, **kwargs) as response:
                return await response.json()

    async def get_model(self):
        data = await self._throttled_request(
            self.session.get,
            self.URL + 'key/api/v1/models'
        )
        return data[0]['id']

    async def generate(self, prompt, model, images=1, width=1024, height=1024):
        params = {
            "type": "GENERATE",
            "numImages": int(images),
            "width": int(width),
            "height": int(height),
            "negativePromptDecoder": "яркие цвета, кислотность, высокая контрастность",
            "generateParams": {
                "query": str(prompt)
            }
        }

        data = FormData()
        data.add_field('model_id', str(model))
        data.add_field('params', json.dumps(params), content_type='application/json')

        response_data = await self._throttled_request(
            self.session.post,
            self.URL + 'key/api/v1/text2image/run',
            data=data
        )
        return response_data.get('uuid')

    async def check_generation(self, request_id, attempts=20, delay=5):
        while attempts > 0:
            data = await self._throttled_request(
                self.session.get,
                self.URL + 'key/api/v1/text2image/status/' + request_id
            )
            
            if data['status'] == 'DONE':
                return data.get('images')
            attempts -= 1
            await asyncio.sleep(delay)
        return None

    async def close(self):
        await self.session.close()

async def save_image(image_base64, output_dir="output", key_prefix=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    unique_id = uuid.uuid4().hex[:8]
    filename = f"image_{int(time.time())}_{key_prefix}_{unique_id}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    image_data = base64.b64decode(image_base64)
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    return filepath

async def worker(prompt, key_pair, total_images, output_dir, progress_tracker):
    token, key = key_pair.split(':')
    api = Text2ImageAPI('https://api-key.fusionbrain.ai/', token, key)
    key_prefix = token[:6]
    
    try:
        model_id = await api.get_model()
        if not model_id:
            print(f"Key {key_prefix}...: Error getting model")
            return

        generated = 0
        while generated < total_images:
            try:
                gen_uuid = await api.generate(prompt, model_id)
                if not gen_uuid:
                    print(f"Key {key_prefix}...: Failed to start generation")
                    continue
                
                images = await api.check_generation(gen_uuid)
                if images:
                    image_path = await save_image(images[0], output_dir, key_prefix)
                    generated += 1
                    await progress_tracker.update(key_prefix, generated, total_images)
                else:
                    print(f"\nKey {key_prefix}...: Generation failed, retrying...")
            except Exception as e:
                print(f"\nKey {key_prefix}...: Error - {str(e)}")
                await asyncio.sleep(2)
    finally:
        await api.close()

async def main():
    prompt = "Девушка курьер на электросамокате в красной повседневной одежде (свитшот, джинсы) с термокоробом за спиной, с телефоном, смотрит в камеру с дружелюбной улыбкой, фон — город, солнечный день, цветение сакуры, стиль — аниме с элементами реализма"
    total_images_per_key = 100
    output_dir = "output"
    
    try:
        with open('keys.txt', 'r') as f:
            key_pairs = [line.strip() for line in f if line.strip() and ':' in line]
    except FileNotFoundError:
        print("Error: keys.txt file not found")
        return

    progress_tracker = ProgressTracker(total_images_per_key, len(key_pairs))
    
    global_semaphore = asyncio.Semaphore(20)
    
    async def throttled_worker(prompt, key_pair, total_images, output_dir, progress_tracker):
        async with global_semaphore:
            await worker(prompt, key_pair, total_images, output_dir, progress_tracker)
            await asyncio.sleep(0.1)
    
    workers = [throttled_worker(prompt, key_pair, total_images_per_key, output_dir, progress_tracker) 
            for key_pair in key_pairs]
    
    await asyncio.gather(*workers)
    progress_tracker.close()

if __name__ == '__main__':
    asyncio.run(main())