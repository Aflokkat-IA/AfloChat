from io import BytesIO
from typing import Any, List, cast

import requests
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from IPython.display import display
from peft import LoraConfig
from PIL import Image
from transformers.models.qwen2_vl import (Qwen2VLForConditionalGeneration,
                                          Qwen2VLProcessor)


def load_image_from_url(url: str) -> Image.Image:
    """
    Load a PIL image from a valid URL.
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    scaled_image = image.resize((new_width, new_height))

    return scaled_image


class ColQwen2ForRAG(ColQwen2):
    """
    ColQwen2 model implementation that can be used both for retrieval and generation.
    Allows switching between retrieval and generation modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_retrieval_enabled = True

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass that calls either Qwen2VLForConditionalGeneration.forward for generation
        or ColQwen2.forward for retrieval based on the current mode.
        """
        if self.is_retrieval_enabled:
            return ColQwen2.forward(self, *args, **kwargs)
        else:
            return Qwen2VLForConditionalGeneration.forward(self, *args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generate text using Qwen2VLForConditionalGeneration.generate.
        """
        if not self.is_generation_enabled:
            raise ValueError(
                "Set the model to generation mode by calling `enable_generation()` before calling `generate()`."
            )
        return super().generate(*args, **kwargs)

    @property
    def is_retrieval_enabled(self) -> bool:
        return self._is_retrieval_enabled

    @property
    def is_generation_enabled(self) -> bool:
        return not self.is_retrieval_enabled

    def enable_retrieval(self) -> None:
        """
        Switch to retrieval mode.
        """
        self.enable_adapters()
        self._is_retrieval_enabled = True

    def enable_generation(self) -> None:
        """
        Switch to generation mode.
        """
        self.disable_adapters()
        self._is_retrieval_enabled = False


model_name = "vidore/colqwen2-v1.0"
device = get_torch_device("auto")

print(f"Using device: {device}")

# Get the LoRA config from the pretrained retrieval model
lora_config = LoraConfig.from_pretrained(model_name)

# Load the processors
processor_retrieval = cast(
    ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))
processor_generation = cast(Qwen2VLProcessor, Qwen2VLProcessor.from_pretrained(
    lora_config.base_model_name_or_path))

# Load the model with the loaded pre-trained adapter for retrieval
model = cast(
    ColQwen2ForRAG,
    ColQwen2ForRAG.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ),
)


# Inputs
query = "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?"

images: List[Image.Image] = [
    load_image_from_url(
        "https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/data/energy_electricity_generation.jpg?raw=true"
    ),
    load_image_from_url(
        "https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/data/shift_kazakhstan.jpg?raw=true"
    ),
]

# NOTE: Because ColQWen2 uses dynamic resolution, we will scale down the images to prevent VRAM overload and faster
# inference times for both indexing and generation. From my experiments, a scale of 512 pixels is a good default for
# document tasks. Feel free to experiment with higher resolutions, especially if the text on your document is small.
images = [scale_image(image, new_height=512) for image in images]
for image in images:
    display(scale_image(image, new_height=256))

# Process the inputs
batch_images = processor_retrieval.process_images(images).to(model.device)
batch_queries = processor_retrieval.process_queries([query]).to(model.device)

# Forward pass
model.enable_retrieval()

with torch.no_grad():
    image_embeddings = model.forward(**batch_images)
    query_embeddings = model.forward(**batch_queries)

# Calculate the similarity scores
scores = processor_retrieval.score_multi_vector(
    query_embeddings, image_embeddings)

# Get the top-1 page image
retrieved_image_index = scores.argmax().item()
retrieved_image = images[retrieved_image_index]

print(f"Image retrieved for the following query: `{query}`")

# Preprocess the inputs
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": f"Answer the following question using the input image: {query}",
            },
        ],
    }
]
text_prompt = processor_generation.apply_chat_template(
    conversation, add_generation_prompt=True)
inputs_generation = processor_generation(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt",
).to(device)

# Generate the RAG response
model.enable_generation()
output_ids = model.generate(**inputs_generation, max_new_tokens=128)

# Ensure that only the newly generated token IDs are retained from output_ids
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(
    inputs_generation.input_ids, output_ids)]

# Decode the RAG response
output_text = processor_generation.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

print(output_text)
