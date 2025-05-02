# Stable Diffusion XL Long Prompt Weighting (LPW) Pipeline

This repository provides an **SDXL-compatible pipeline** that enables:
- **Unlimited prompt length** (bypassing the 77-token CLIP limit)
- **Prompt weighting** using familiar `(text:weight)` and `[text]` syntax
- **Drop-in replacement** for `diffusers` Stable Diffusion XL pipelines

---

## ✨ Features

- **Arbitrary prompt length:** Prompts are split into 75-token chunks, encoded, and their embeddings concatenated.
- **Prompt weighting:** Use `(word:1.5)` to increase, `[word]` to decrease, or nested parentheses for more emphasis.
- **Supports all SDXL features:** Works with ControlNet, Inpainting, Img2Img, LoRA, IP-Adapter, etc.
- **No prompt truncation:** All of your prompt is used, regardless of length.

---

## 🚀 Usage

Just like a regular `diffusers` pipeline:

```python
from lpw_stable_diffusion_xl import StableDiffusionXLLongPromptPipeline

pipe = StableDiffusionXLLongPromptPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16
).to("cuda")

prompt = "A (highly detailed:1.5) futuristic cityscape with (flying cars:1.2), (neon lights:1.3), [smoggy sky], " +
"and a bustling crowd of people in unique attire, " +
"with towering buildings stretching into the clouds. " * 10 # Make prompt very long

image = pipe(prompt).images
image.save("output.png")
```

### Prompt Weighting Syntax

- `(word)` or `(word:1.1)`: Increase emphasis (default factor 1.1)
- `(word:2.0)`: Strongly increase emphasis (factor 2.0)
- `[word]`: Decrease emphasis (factor 1/1.1)
- Nesting: `((word))` increases emphasis further

---

## 🧠 How It Works

### The CLIP Token Limit

CLIP text encoders (used in SDXL) can only process **77 tokens at once**. Normally, longer prompts are truncated, losing detail.

### The Solution: Chunking & Concatenation

1. **Tokenization & Weight Parsing:**  
   The prompt is tokenized and parsed for weighting syntax. Each token is assigned a weight.

2. **Chunking:**  
   Tokens are split into groups of 75, with BOS/EOS tokens added to each chunk to form 77-token segments.

3. **Encoding:**  
   Each chunk is encoded independently by the CLIP text encoder(s).

4. **Weight Application:**  
   The parsed weights are applied directly to the token embeddings.

5. **Concatenation:**  
   All chunk embeddings are concatenated to form a single, long embedding representing the whole prompt.

6. **Image Generation:**  
   This embedding is used in the diffusion process, so the model "sees" the entire prompt, not just the first 77 tokens.

### Integration

The pipeline replaces the default prompt encoding:

```python
(
prompt_embeds,
negative_prompt_embeds,
pooled_prompt_embeds,
negative_pooled_prompt_embeds,
) = lpw.get_weighted_text_embeddings_sdxl(
pipe=self,
prompt=prompt,
neg_prompt=negative_prompt,
prompt_embeds=prompt_embeds,
negative_prompt_embeds=negative_prompt_embeds,
pooled_prompt_embeds=pooled_prompt_embeds,
negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
)
```

---

## ⚠️ Limitations

- **Memory:** Very long prompts use more GPU memory.
- **Prompt balancing:** Extreme weighting can produce unpredictable results.
- **Performance:** Slightly slower with very long prompts.

---

## 📝 Example: ControlNet & Img2Img

```python
from diffusers import ControlNetModel
from lpw_stable_diffusion_xl import StableDiffusionXLLongPromptPipeline

controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")
pipe = StableDiffusionXLLongPromptPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
controlnet=controlnet,
torch_dtype=torch.float16
).to("cuda")

prompt = "(robot:1.5) in a futuristic laboratory, [messy background], " + "with advanced equipment " * 20
image = pipe(prompt, control_image=canny_edges).images
```

---

## 📚 References

- [HuggingFace Diffusers Long Prompt Weighting Example](https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion_xl.py)
- [Stable Diffusion XL Model Card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Common Diffusion Noise Schedules and Sample Steps are Flawed (arXiv:2305.08891)](https://arxiv.org/pdf/2305.08891.pdf)

---

## 📄 License

Apache 2.0 - Same as Stable Diffusion XL base model.

---

## 🙋 FAQ

**Q:** How long can my prompt be?  
**A:** As long as you want! The pipeline splits and encodes it in chunks.

**Q:** Is this compatible with LoRA, ControlNet, etc.?  
**A:** Yes, it's a drop-in replacement for SDXL pipelines.

**Q:** Does prompt weighting work for negative prompts?  
**A:** Yes, weighting syntax is supported for both positive and negative prompts.

---

Enjoy unlimited, expressive prompting in SDXL!
