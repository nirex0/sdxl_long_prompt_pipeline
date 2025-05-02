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

Pick one of these prompts and substitute of the prompt variable:

Here are several examples of long, detailed prompts suitable for Stable Diffusion XL or other text-to-image models. Each prompt is at least a paragraph in length and describes a scene with rich detail, using explicit sensory and visual cues, as recommended for effective AI image generation.

---

**1. Enchanted Forest at Dawn**

A sprawling, ancient forest bathed in the soft golden light of early morning, with towering trees covered in thick moss and delicate ferns carpeting the forest floor. Shafts of sunlight break through the dense canopy, illuminating patches of wildflowers in vibrant colors and casting intricate shadows. In the distance, a gentle mist hovers above a crystal-clear stream winding through the trees, reflecting the pastel hues of the sky. The air is filled with the subtle glow of fireflies, and a family of deer grazes quietly near the water’s edge, while birds in brilliant plumage flit between branches, their songs echoing through the tranquil scene.

---

**2. Futuristic Cityscape at Night**

A bustling metropolis in the heart of a futuristic world, where towering skyscrapers of glass and steel are adorned with glowing neon signs in every color imaginable. The streets below are alive with streams of autonomous vehicles, their headlights creating ribbons of light that weave through the city. Crowds of people in avant-garde fashion move along elevated walkways, while holographic advertisements flicker above them, casting a surreal glow. In the distance, a monorail glides silently past a shimmering river that reflects the dazzling city lights, and the sky is filled with flying cars darting between the buildings under a canopy of artificial stars.

---

**3. Cozy Mountain Cabin in Winter**

Nestled in a snow-covered valley surrounded by towering pine trees, a rustic wooden cabin emits a warm, inviting glow from its windows. Smoke curls lazily from the stone chimney, hinting at a roaring fire within. The landscape is blanketed in fresh, untouched snow that sparkles under the pale light of a full moon. Icicles hang from the eaves, and a trail of footprints leads from the front porch to a frozen pond nearby, where children in colorful scarves skate and laugh. The crisp night air is filled with the scent of pine and the distant sound of an owl hooting, creating a serene and magical winter scene.

---

**4. Bustling Market in Marrakesh**

A vibrant open-air market in Marrakesh, alive with a tapestry of colors, sounds, and scents. Stalls overflowing with spices, fruits, and textiles line the narrow, winding alleys, their wares displayed in intricate patterns. The air is thick with the aroma of saffron, cinnamon, and roasting meats, while vendors call out to passersby in a lively mix of languages. Sunlight filters through ornate latticework, casting decorative shadows on the cobblestone streets. Shoppers in traditional and modern attire haggle over handwoven rugs and gleaming brass lanterns, and musicians play rhythmic melodies on stringed instruments, adding to the energetic, festive atmosphere.

---

**5. Serene Japanese Garden in Spring**

A tranquil Japanese garden in full bloom during spring, with delicate cherry blossoms drifting gently to the ground and forming a soft pink carpet around a peaceful koi pond. Stone lanterns and arched wooden bridges span the water, where brightly colored koi swim lazily beneath the surface. Manicured shrubs and moss-covered rocks create a harmonious landscape, while a gentle breeze carries the scent of blooming wisteria. In the background, a traditional tea house with sliding shoji doors overlooks the garden, and a pair of cranes stand gracefully at the water’s edge, completing the scene of quiet beauty and balance.

---

Just like a regular `diffusers` pipeline:


```python
from lpw_stable_diffusion_xl import StableDiffusionXLLongPromptPipeline

pipe = StableDiffusionXLLongPromptPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16
).to("cuda")

prompt = "A tranquil Japanese garden in full bloom during spring, with delicate cherry blossoms drifting gently to the ground and forming a soft pink carpet around a peaceful koi pond. Stone lanterns and arched wooden bridges span the water, where brightly colored koi swim lazily beneath the surface. Manicured shrubs and moss-covered rocks create a harmonious landscape, while a gentle breeze carries the scent of blooming wisteria. In the background, a traditional tea house with sliding shoji doors overlooks the garden, and a pair of cranes stand gracefully at the water’s edge, completing the scene of quiet beauty and balance."

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
