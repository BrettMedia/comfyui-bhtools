# comfyui-bhtools/nodes/CinematicSceneDirectorNodeBHTools.py
print("DEBUG: CinematicSceneDirectorDirectorTools.py is being processed.") # Diagnostic print

import random
import os
import json
import gc
import torch
import re

# Check for transformers availability
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        BitsAndBytesConfig, GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("\n--- WARNING: 'transformers' library not found. LLM inference will use fallback enhancement. ---")
    print("--- Please install it with: pip install transformers torch accelerate bitsandbytes sentencepiece ---\n")

# Check for bitsandbytes availability for quantization
BITSANDBYTES_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    pass


class CinematicSceneDirectorTools:
    """
    Advanced Cinematic Scene Director Node for ComfyUI
    
    Creates sophisticated cinematic prompts by cycling through master prompts
    and applying comprehensive cinematic parameters including composition,
    lighting, atmosphere, and post-production effects.
    
    Features:
    - Master prompts system with auto-incrementing
    - Comprehensive cinematic controls
    - Weighted parameter system
    - Preset overrides for quick styling
    - Optional LLM-powered prompt enhancement with memory management.
    - Animation specific controls.
    - Delineated list generation based on a subject.
    - Contextual help information.
    - Collapsible sections for economical design.
    """
    
    # Class-level definition of PRESETS
    PRESETS = {
        "cinematic_preset": {
            "Cinematic Drama": "visually striking cinematic scene with deep shadows, rich contrast, and dynamic composition",
            "Epic Cinematic": "grand sweeping cinematic vista with dramatic scale and breathtaking composition",
            "Intimate Drama": "close personal cinematic moment with subtle lighting and emotional depth",
            "Blockbuster Action": "high-energy cinematic action scene with dynamic movement and bold lighting",
            "Film Noir": "classic film noir style, high contrast, stark shadows, dramatic lighting",
            "Gritty Realism": "raw, unpolished, authentic, documentary-like realism, natural lighting",
            "Dreamy Sequence": "soft focus, ethereal lighting, hazy atmosphere, surreal elements",
            "Suspense Thriller": "tense atmosphere, low-key lighting, unsettling shadows, psychological depth",
            "Neo-Noir": "modern film noir, vibrant city lights, moral ambiguity, stylish crime",
            "Fantasy Adventure": "heroic journey, mythical creatures, ancient ruins, epic landscapes",
            "Sci-Fi Odyssey": "interstellar travel, advanced AI, cosmic phenomena, existential themes",
            "Historical Epic": "sweeping historical events, period costumes, grand battles, political intrigue",
            "Musical Extravaganza": "vibrant musical numbers, elaborate choreography, dazzling costumes, theatrical lighting",
            "Heist Thriller": "intricate planning, high stakes, tense execution, slick visuals, cunning characters",
            "Coming-of-Age Drama": "youthful discovery, emotional growth, nostalgic setting, relatable struggles",
            "Post-Apocalyptic Survival": "desolate landscapes, desperate struggle, makeshift shelters, resilient survivors",
            "Cyberpunk Dystopia": "futuristic city, corporate control, neon glow, augmented humans, gritty underworld",
            "Western Showdown": "dusty plains, lone rider, tense standoff, classic frontier town, dramatic sunset",
            "Martial Arts Epic": "fluid combat, graceful movements, ancient traditions, serene landscapes, powerful strikes",
            "Space Western": "frontier justice in space, rugged spacecraft, alien outlaws, dusty red planets",
            "Underwater Exploration": "mysterious deep sea, bioluminescent creatures, ancient ruins submerged, ethereal light",
            "Arctic Expedition": "frozen wilderness, biting winds, survival against elements, vast icy landscapes",
            "Jungle Expedition": "dense jungle, hidden temples, exotic wildlife, humid atmosphere, ancient mysteries",
            "Desert Nomad": "endless dunes, scorching sun, resilient travelers, ancient secrets buried in sand",
            "Volcanic Inferno": "fiery landscapes, molten rivers, ash-filled skies, extreme heat, dangerous environment",
        },
        "lighting_preset": {
            "Golden Hour Magic": "warm golden hour cinematic scene with soft directional lighting and magical atmosphere",
            "Blue Hour Mystery": "mysterious blue hour cinematic scene with cool tones and atmospheric depth",
            "Noir Shadows": "film noir cinematic scene with stark contrasts and dramatic shadow play",
            "High Key Bright": "bright high-key cinematic scene with even lighting and minimal shadows",
            "Dramatic Backlighting": "strong backlighting, rim light, silhouette, dramatic contrast",
            "Soft Studio Lighting": "even, diffused lighting, soft shadows, professional studio setup",
            "Hard Industrial Light": "harsh, direct light, strong shadows, industrial setting, utilitarian feel",
            "Candlelit Glow": "warm, flickering light, soft shadows, intimate and cozy atmosphere",
            "Neon City Lights": "vibrant neon glow, urban night, cyberpunk aesthetic, reflective surfaces",
            "Volumetric Fog Lighting": "light rays cutting through dense fog, ethereal, mystical, atmospheric depth",
            "Moonlit Serenity": "soft moonlight, tranquil, cool tones, peaceful night scene",
            "Fluorescent Hum": "harsh, sterile fluorescent lighting, institutional, unsettling atmosphere",
            "Spotlight Drama": "intense single spotlight, deep shadows, theatrical, focused attention",
            "Rim Light Emphasis": "strong rim light, separates subject from background, creates depth, dramatic contour",
            "Chiaroscuro Effect": "strong contrasts between light and dark, dramatic illumination, painterly quality",
            "Ambient Glow": "soft, diffused light, fills the scene evenly, gentle, calming atmosphere",
            "Dappled Sunlight": "sunlight filtering through leaves, patchy light and shadow, natural, serene",
            "Underwater Caustics": "rippling light patterns from water surface, ethereal, aquatic, dynamic",
            "Firelight Flicker": "dynamic flickering light, warm tones, creates movement and intimacy",
            "Strobe Flash": "intermittent, harsh flashes of light, disorienting, intense, chaotic",
            "Gel Filtered Light": "colored light, mood-setting, artistic, vibrant, atmospheric tint",
            "Natural Window Light": "soft, directional light from a window, realistic, intimate, cozy",
            "Overhead Harsh Light": "direct light from above, strong shadows below, dramatic, oppressive",
            "Silhouetted Figures": "subjects as dark shapes against bright background, mysterious, dramatic, iconic",
            "Concert Lighting": "dynamic, colorful stage lights, spotlights, haze, energetic atmosphere",
        },
        "genre_preset": {
            "Sci-Fi Futuristic": "futuristic sci-fi cinematic scene with advanced technology and sleek aesthetics",
            "Fantasy Epic": "epic fantasy cinematic scene with magical elements and otherworldly beauty",
            "Horror Atmospheric": "atmospheric horror cinematic scene with eerie lighting and unsettling mood",
            "Romance Dreamy": "romantic dreamy cinematic scene with soft lighting and warm emotional tones",
            "Western Gritty": "gritty western cinematic scene with dusty atmosphere and harsh lighting",
            "Cyberpunk Neon": "cyberpunk cinematic scene with neon lighting and urban futuristic elements",
            "Period Historical": "historically accurate cinematic scene with period-appropriate styling and atmosphere",
            "Documentary Real": "realistic documentary-style cinematic scene with natural lighting and authentic feel",
            "Steampunk Aesthetic": "steampunk style, brass, gears, Victorian era, intricate machinery",
            "Post-Apocalyptic": "desolate, ruined landscapes, survivalist aesthetic, muted colors",
            "Utopian Vision": "bright, clean, harmonious, futuristic, ideal society aesthetic",
            "Dystopian Future": "oppressive, grim, totalitarian, decaying urban environment",
            "Mythological Saga": "ancient, heroic, grand scale, legendary figures, epic narrative",
            "Urban Fantasy": "magical elements hidden within a contemporary city, subtle enchantment",
            "Magical Realism": "ordinary setting with fantastical elements treated as commonplace",
            "Space Opera": "grand scale space battles, alien civilizations, galactic empires",
            "Superhero Action": "dynamic superhero poses, city destruction, powerful abilities, comic book style",
            "Spy Thriller": "covert operations, double agents, high-tech gadgets, international intrigue",
            "Legal Drama": "courtroom tension, moral dilemmas, legal battles, compelling arguments",
            "Medical Drama": "hospital setting, life-or-death situations, medical mysteries, human stories",
            "Sports Drama": "athletic competition, underdog story, intense training, triumphant victory",
            "Disaster Film": "cataclysmic events, widespread destruction, human resilience, survival",
            "Biographical Film": "life story of a historical figure, personal journey, significant achievements",
            "Road Movie": "journey of self-discovery, diverse landscapes, unexpected encounters, personal transformation",
            "Conspiracy Thriller": "secret organizations, hidden truths, paranoia, pursuit of answers",
        },
        "visual_style_preset": {
            "Oil Painting Masterpiece": "rich oil painting, visible brushstrokes, classic art style, vibrant colors",
            "Watercolor Illustration": "fluid watercolor, soft edges, translucent washes, illustrative style",
            "Pencil Sketch Art": "detailed pencil sketch, cross-hatching, monochromatic, hand-drawn quality",
            "Digital Painting": "smooth digital painting, crisp lines, vibrant colors, clean rendering",
            "Concept Art Style": "dynamic concept art, imaginative design, detailed world-building, illustrative",
            "Anime Aesthetic": "anime style, expressive characters, vibrant colors, dynamic poses, cel-shaded",
            "Cartoon Style": "cartoon style, bold outlines, simplified forms, exaggerated features, bright colors",
            "Pixel Art": "retro pixel art, low resolution, blocky aesthetic, nostalgic charm",
            "Vector Art": "clean vector art, sharp edges, flat colors, minimalist design",
            "Abstract Expressionism": "abstract expressionist painting, bold brushstrokes, emotional, non-representational",
            "Surrealist Dream": "surrealist art, dreamlike, illogical juxtapositions, symbolic imagery",
            "Impressionistic Brushstrokes": "impressionistic painting, soft focus, visible brushstrokes, light and color emphasis",
            "Pop Art Vibrant": "pop art style, bold colors, graphic lines, mass culture references",
            "Art Deco Elegance": "art deco style, geometric patterns, luxurious, symmetrical, elegant",
            "Minimalist Clean": "minimalist design, clean lines, ample negative space, simple forms",
            "Maximalist Rich": "maximalist design, rich patterns, layered textures, abundant details",
            "Retro Nostalgic": "nostalgic retro aesthetic, vintage colors, classic design elements",
            "Avant Garde": "avant-garde style, experimental, unconventional, pushing artistic boundaries",
            "Cybernetic Art": "fusion of organic and mechanical, glowing circuits, futuristic biomechanics",
            "Steampunk Illustration": "intricate gears, Victorian machinery, brass and copper tones, elaborate contraptions",
            "Photorealism": "ultra-realistic, lifelike detail, indistinguishable from a photograph",
            "Low Poly": "geometric, faceted, simplified forms, retro 3D aesthetic",
            "Vaporwave Aesthetic": "neon colors, glitch effects, classical sculptures, retrofuturism",
            "Gothic Revival": "dark, ornate, dramatic, medieval influences, intricate details",
            "Baroque Opulence": "rich, elaborate, dramatic, highly detailed, dynamic compositions",
            "Rococo Charm": "light, playful, ornate, pastel colors, asymmetrical designs",
            "Art Nouveau Flow": "organic lines, natural forms, flowing curves, decorative, elegant",
            "Bauhaus Simplicity": "functional, minimalist, geometric, industrial, clean lines",
            "Street Art Mural": "bold colors, graphic style, urban environment, spray paint textures",
            "Comic Book Style": "inked outlines, halftone dots, dynamic action, speech bubbles",
        },
        "vfx_preset": {
            "Explosion VFX": "dynamic explosion, fiery blast, smoke and debris, impactful visual effects",
            "Energy Blast VFX": "powerful energy blast, glowing particles, light trails, magical or sci-fi effect",
            "Smoke & Fire VFX": "realistic smoke and fire simulation, dynamic flames, drifting smoke",
            "Water Simulation VFX": "realistic water simulation, splashing, flowing, rippling water effects",
            "Magical Aura VFX": "glowing magical aura, shimmering light, ethereal particles, fantasy effect",
            "Particle System VFX": "complex particle system, swirling dust, glittering effects, abstract patterns",
            "Glitch Effect VFX": "digital glitch effect, distorted pixels, color shifting, broken signal aesthetic",
            "Holographic Display VFX": "holographic display, glowing projections, futuristic UI, transparent elements",
            "Force Field VFX": "shimmering force field, energy distortion, protective barrier effect",
            "Teleportation Effect": "disintegrating and reforming particles, light trails, instantaneous travel effect",
            "Electric Discharge VFX": "cracking electrical discharge, lightning bolts, arcs of energy",
            "Time Distortion": "warped reality, slow motion, fast forward, temporal anomalies",
            "Dimensional Rift": "tearing fabric of space, swirling vortex, otherworldly portal",
            "Gravity Manipulation": "objects floating, distorted space, controlled levitation, zero-g environment",
            "Invisibility Cloak": "shimmering distortion, partial transparency, camouflaged figures, subtle effect",
            "Healing Glow": "soft, warm light, radiating energy, wounds closing, rejuvenating effect",
            "Destruction Physics": "realistic debris, crumbling structures, impact craters, dynamic collapse",
            "Weather Control": "instantaneous storms, localized blizzards, sudden droughts, atmospheric manipulation",
            "Mind Control Aura": "subtle shimmering around eyes, hypnotic influence, visible thought waves",
            "Sound Wave Visualizer": "vibrating air, visible sound patterns, rhythmic pulses, sonic energy",
            "Portal Opening": "swirling energy, expanding gateway, light emanating, transition to another world",
            "Energy Shield": "translucent barrier, deflecting attacks, glowing edges, protective dome",
        },
        "domain_specific_preset": {
            "Character Portrait": "detailed character portrait, expressive features, focused lighting, personality driven",
            "Fashion Photography": "high fashion photography, elegant poses, dramatic lighting, stylish attire",
            "Product Photography": "clean product photography, sharp focus, ideal lighting, isolated background",
            "Architectural Render": "photorealistic architectural render, detailed building, accurate lighting, grand scale",
            "Landscape Photography": "expansive landscape photography, natural light, scenic vista, deep focus",
            "Macro Photography": "extreme macro photography, intricate details, shallow depth of field, close-up view",
            "Street Photography": "candid street photography, urban environment, natural light, capturing everyday life",
            "Wildlife Photography": "authentic wildlife photography, natural habitat, sharp detail, animal in action",
            "Food Photography": "appetizing food photography, appealing presentation, warm lighting, delicious textures",
            "Automotive Photography": "sleek automotive photography, dynamic angles, reflective surfaces, powerful stance",
            "Sports Photography": "action sports photography, frozen motion, dynamic composition, high energy",
            "Logo Design Minimalist": "minimalist logo design, clean lines, simple shapes, effective branding",
            "Logo Design Modern": "modern logo design, sleek typography, contemporary aesthetic, digital feel",
            "Logo Design Vintage": "vintage logo design, retro typography, classic elements, nostalgic appeal",
            "Book Cover Illustration": "engaging book cover illustration, genre-appropriate, compelling imagery",
            "Album Art Design": "creative album art design, visually striking, mood-setting, musical theme",
            "Game Asset Design": "detailed game asset design, optimized for engine, consistent style, functional aesthetics",
            "Infographic Style": "clear infographic style, data visualization, illustrative icons, informative layout",
            "Medical Illustration": "precise medical illustration, anatomical accuracy, scientific detail",
            "Botanical Illustration": "detailed botanical illustration, scientific accuracy, natural textures, vibrant colors",
        },
        "environment_preset": {
            "Urban Cityscape": "urban cityscape cinematic scene with metropolitan atmosphere and architectural elements",
            "Natural Landscape": "natural landscape cinematic scene with organic beauty and environmental depth",
            "Interior Intimate": "intimate interior cinematic scene with controlled lighting and personal atmosphere", # Added Interior preset
            "Cosmic Space": "cosmic space cinematic scene with stellar elements and infinite depth",
            "Underwater Depths": "underwater cinematic scene with aquatic lighting and fluid movement",
            "Desert Vastness": "vast desert cinematic scene with expansive horizons and harsh beauty",
            "Forest Mystical": "mystical forest cinematic scene with dappled light and natural mystery",
            "Mountain Majestic": "majestic mountain cinematic scene with dramatic elevation and natural grandeur",
            "Arctic Tundra": "frozen arctic tundra, vast icy plains, cold atmosphere, stark beauty",
            "Tropical Jungle": "dense tropical jungle, lush vegetation, humid atmosphere, vibrant colors",
            "Volcanic Landscape": "dramatic volcanic landscape, fiery glow, harsh terrain, smoky atmosphere",
            "Canyon Vista": "sweeping canyon vista, red rock formations, vast open spaces, dramatic shadows",
            "Ancient Ruins": "ancient ruins, overgrown structures, historical atmosphere, weathered stone",
            "Futuristic City": "futoring skyscrapers, flying vehicles, advanced technology",
            "Rural Farmland": "peaceful rural farmland, rolling hills, agricultural landscape, rustic charm",
            "Seaside Village": "charming seaside village, coastal architecture, ocean views, tranquil atmosphere",
            "Subterranean Caverns": "dark, echoing caverns, glowing crystals, hidden ancient structures",
            "Floating Islands": "islands suspended in sky, waterfalls cascading into clouds, ethereal landscape",
        },
        "humor_preset": {
            "Slapstick Comedy": "exaggerated, physical comedy, cartoonish, lighthearted, humorous",
            "Dark Humor": "sarcastic, cynical, morbidly funny, satirical, witty",
            "Absurdist Comedy": "surreal, illogical, nonsensical, bizarre, whimsical",
            "Romantic Comedy": "charming, lighthearted, witty dialogue, heartwarming, playful",
            "Satirical Comedy": "social commentary, ironic, sharp wit, political humor",
            "Parody Style": "mimics and exaggerates, comedic imitation, recognizable tropes",
            "Situational Comedy": "everyday situations, relatable characters, comedic misunderstandings",
            "Surreal Humor": "dreamlike, unexpected twists, unconventional narratives, bizarre elements",
            "Self-Referential Humor": "breaks the fourth wall, acknowledges its own fictionality, meta-comedy",
            "Character-Driven Comedy": "humor from character quirks, relatable flaws, personality clashes",
            "Farcical Comedy": "exaggerated situations, improbable events, rapid-fire dialogue, chaotic fun",
            "Observational Comedy": "humor from everyday life, relatable experiences, keen social insight",
        },
        "horror_preset": {
            "Psychological Horror": "tense, unsettling, suspenseful, mind-bending, disturbing",
            "Gothic Horror": "eerie, atmospheric, supernatural, Victorian era, mysterious",
            "Body Horror": "visceral, grotesque, disfigured, biological, unsettling",
            "Found Footage Horror": "raw, shaky cam, realistic, first-person perspective, immersive",
            "Cosmic Horror": "eldritch, vast, incomprehensible, existential dread, ancient evil",
            "Slasher Film": "masked killer, suspenseful chase, gory, classic horror tropes",
            "Zombie Apocalypse": "undead hordes, desperate survival, decaying world, gruesome",
            "Supernatural Thriller": "ghosts, demons, cursed objects, paranormal activity, jump scares",
            "Folk Horror": "ancient rituals, isolated communities, paganism, unsettling traditions",
            "Techno-Horror": "malicious AI, digital threats, technological dystopia, surveillance",
            "Survival Horror": "resource management, limited combat, oppressive atmosphere, desperate escape",
            "Creature Feature": "monstrous beings, terrifying beasts, primal fear, destructive rampage",
        },
        "nsfw_preset": { # Non-explicit, suggestive themes
            "Dark Themes": "mature, somber, intense, thought-provoking, adult narrative",
            "Gritty Realism (Adult)": "raw, unvarnished, authentic, unflinching, mature content implied",
            "Sensual Atmosphere": "intimate, suggestive, alluring, romantic, artistic nudity implied",
            "Noir Detective (Adult)": "crime, mystery, femme fatale, shadowy, morally ambiguous",
            "Post-Apocalyptic (Adult)": "survival, desolation, harsh realities, mature themes of struggle",
            "Forbidden Romance": "clandestine, passionate, illicit, intense emotional connection",
            "Taboo Subject": "controversial, provocative, explores societal norms, thought-provoking",
            "Erotic Fantasy": "mythical, fantastical, alluring, enchanting, dreamlike",
            "Adult Animation Style": "stylized, vibrant, mature themes, distinct visual aesthetic",
            "Urban Underbelly": "gritty cityscapes, hidden desires, dark secrets, nocturnal activities",
            "Vampire Seduction": "alluring vampires, gothic romance, dark desires, immortal passion",
            "Steamy Sci-Fi": "futuristic settings, advanced technology, sensual encounters, cybernetic allure",
            "Not For Mom": "beautiful, erotic, naked, sensual, seductive, nipples, breasts, vagina, posed suggestivey, alluring, pornographic, nsfw",
        },
        "hentai_preset": { # Stylistic and thematic elements, non-explicit
            "Anime Art Style": "anime art style, vibrant colors, expressive lines, dynamic poses",
            "Manga Aesthetic": "manga aesthetic, black and white, detailed line art, dramatic shading",
            "Fantasy Creatures": "fantasy creatures, mythical beings, magical elements, fantastical setting",
            "Futuristic Setting": "futuristic setting, cybernetic elements, advanced technology, neon lights",
            "School Uniform Theme": "school uniform theme, academic setting, youthful characters",
            "Magical Girl Theme": "magical girl theme, transformation sequences, colorful powers, whimsical elements",
            "Mythological Figures": "mythological figures, ancient deities, legendary beings, epic scale",
            "Cybernetic Enhancements": "cybernetic enhancements, robotic parts, human-machine fusion, futuristic design",
            "Kemonomimi Characters": "animal ears, tails, human-animal hybrid characters, cute, playful",
            "Ecchi Elements": "lighthearted, playful, fan service, suggestive but not explicit",
            "Chibi Style": "cute, small, exaggerated heads, simplified forms",
            "Mecha Integration": "giant robots, mechanical suits, human pilots, futuristic combat",
            "Alien Perv": "cute, tentacles, alien sex, masturbation, cosplay, short skirt, small breasts, orgasmic face",
        },
        "character_preset": { # New character preset
            "Heroic Figure": "heroic figure, strong, determined, noble, courageous",
            "Villainous Antagonist": "villainous antagonist, cunning, menacing, powerful, dark aura",
            "Mysterious Stranger": "mysterious stranger, enigmatic, secretive, intriguing, unknown motives",
            "Innocent Child": "innocent child, curious, playful, vulnerable, pure",
            "Wise Elder": "wise elder, serene, experienced, knowledgeable, calm demeanor",
            "Rebellious Youth": "rebellious youth, defiant, energetic, independent, non-conformist",
            "Elegant Aristocrat": "elegant aristocrat, refined, graceful, sophisticated, poised",
            "Rough Mercenary": "rough mercenary, battle-hardened, pragmatic, tough, self-reliant",
            "Fantasy Warrior": "brave warrior, armored, weapon-wielding, battle-ready",
            "Sci-Fi Explorer": "futuristic explorer, space suit, alien environments, adventurous",
            "Cyberpunk Hacker": "tech-savvy hacker, neon-lit, augmented, rebellious",
            "Magical Apprentice": "young magic user, spellcaster, enchanted, learning powers",
        }
    }

    NEGATIVE_PROMPT_PRESETS = {
        "None": "",
        "General Undesirables": "ugly, deformed, noisy, blurry, low resolution, distorted, grainy, bad anatomy, poorly drawn, malformed limbs, missing limbs, extra limbs, fused fingers, too many fingers, watermark, signature, text, error, cropped, out of frame, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutilated, bad proportions, gross proportions, too many eyes, too many heads, bad hands, extra digit, fewer digits, extra arms, extra legs, extra body, extra head, extra face, extra foot, extra feet, extra hand, extra hands, extra finger, extra fingers, extra thumb, extra thumbs, extra toe, extra toes, bad face, bad eyes, bad mouth, bad nose, bad ear, bad hair, bad skin, bad teeth, bad tongue, bad lips, bad chin, bad forehead, bad neck, bad shoulder, bad arm, bad leg, bad foot, bad hand, bad finger, bad thumb, bad toe, bad nail, bad joint, bad bone, bad muscle, bad tendon, bad vein, bad artery, bad nerve, bad gland, bad organ, bad cell, bad tissue, bad blood, bad fluid, bad system, bad body, bad anatomy, bad composition, bad perspective, bad lighting, bad shadow, bad color, bad contrast, bad saturation, bad hue, bad brightness, bad sharpness, bad focus, bad exposure, bad white balance, bad noise, bad grain, bad blur, bad distortion, bad artifact, bad compression, bad filter, bad effect, bad render, bad quality, bad image, bad picture, bad photo, bad drawing, bad painting, bad illustration, bad art, bad creation, bad generation, bad output, bad result, bad anything, bad everything, bad",
        "Artistic Imperfections": "poorly drawn, bad art, amateur, ugly, deformed, disfigured, low quality, pixelated, blurry, noisy, oversaturated, undersaturated, bad composition, watermark, signature, text",
        "Anatomical Errors": "bad anatomy, malformed limbs, missing limbs, extra limbs, fused fingers, too many fingers, too many eyes, too many heads, ugly, deformed, disfigured",
        "Low Quality Output": "low resolution, grainy, jpeg artifacts, blurry, noisy, compressed, pixelated, worst quality, low quality, bad quality",
        "Censorship Avoidance": "blurry, censored, pixelated, distorted, covered, text, watermark, signature, nudity, bare breasts, nipples, penis, vagina, pubic hair, anus, sex, sexual, explicit, porn, pornographic, erotic, suggestive",
        "Style Conflicts": "cartoon, anime, 3D render, illustration, sketch, painting, drawing, comic, sculpture, low poly, pixel art, abstract, text, watermark, signature",
        "Distorted Features": "distorted face, disfigured, malformed, mutated, ugly, blurry, noisy, bad eyes, bad mouth, bad nose, bad ears",
        "Unwanted Elements": "watermark, signature, text, logo, copyright, frame, border, cropped, cut off, out of frame, extra limbs, extra fingers, deformed hands, bad anatomy, low quality, blurry, noisy, bad composition",
        "AI Artifacts": "unrealistic, distorted, bad hands, extra fingers, blurry, noisy, watermark, text, signature, low quality, bad anatomy, deformed, mutated, fused, duplicate, cloned, malformed, disfigured, ugly",
        "Concept Art Undesirables": "unfinished, rough sketch, messy lines, unrefined, low detail, watermark, text, signature, blurry, noisy",
        "Photographic Flaws": "overexposed, underexposed, blurry, noisy, grainy, bad focus, bad lighting, bad color, bad composition, watermark, signature, text, lens flare, chromatic aberration, distortion",
        "Animation Imperfections": "jerky animation, stiff movement, low frame rate, inconsistent style, glitches, artifacts, blurry, noisy",
        "Abstract Undesirables": "clear forms, recognizable objects, realistic, symmetrical, orderly, text, watermark, signature",
    }

    # Define adjustable details options as class variables for easier normalization
    # These are the 'values' lists for the STRING inputs
    ADJUSTABLE_DETAIL_OPTIONS = {
        "shot_type": ["Extreme Wide Shot", "Wide Shot", "Full Shot", "Medium Full Shot",
                      "Medium Shot", "Medium Close-Up", "Close-Up", "Extreme Close-Up",
                      "Two-Shot", "Over-the-Shoulder", "Point of View", "Insert Shot",
                      "Cutaway", "Establishing Shot", "Master Shot", "Dolly Shot", "Tracking Shot", "Zoom Shot"],
        "camera_angle": ["Eye Level", "High Angle", "Low Angle", "Bird's Eye View", "Worm's Eye View",
                         "Dutch Angle", "Overhead Shot", "Aerial View", "Ground Level",
                         "Shoulder Level", "Hip Level", "Knee Level", "Canted Angle", "Point of View (POV)",
                         "Extreme High Angle", "Extreme Low Angle"],
        "camera_movement": ["Static Shot", "Pan Left", "Pan Right", "Tilt Up", "Tilt Down",
                            "Dolly In", "Dolly Out", "Truck Left", "Truck Right", "Pedestal Up", "Pedestal Down",
                            "Zoom In", "Zoom Out", "Rack Focus", "Handheld", "Steadicam", "Gimbal",
                            "Crane Up", "Crane Down", "Jib Arm", "Slider", "Circular Dolly", "Parallax",
                            "Arc Shot", "Follow Shot", "Lead Shot", "Reveal Shot", "Push In", "Pull Out"],
        "camera_lens": ["Ultra Wide 14mm", "Wide Angle 24mm", "Standard 35mm", "Portrait 50mm", 
                        "Short Telephoto 85mm", "Telephoto 135mm", "Super Telephoto 200mm",
                        "Fisheye", "Macro", "Tilt-Shift", "Anamorphic", "Vintage Lens", "Prime Lens", "Zoom Lens",
                        "Spherical Lens", "Soft Focus Lens", "Infrared Lens", "UV Lens", "Periscope Lens"],
        "aspect_ratio": ["16:9 Widescreen", "21:9 Ultrawide", "2.35:1 Anamorphic", "1.85:1 Academy",
                         "4:3 Classic", "1:1 Square", "9:16 Vertical", "3:2 Photography",
                         "5:4 Large Format", "2.39:1 Cinemascope", "1.33:1 Academy Classic",
                         "2.76:1 Ultra Panavision", "Custom Ratio", "1.78:1 HD", "2.20:1 Todd-AO",
                         "1.66:1 European Widescreen", "1.37:1 Academy Standard"],
        "depth_of_field": ["Shallow DOF", "Deep DOF", "Rack Focus", "Split Focus", "Hyperfocal",
                           "Bokeh Heavy", "Tilt-Shift Miniature", "Focus Pulling", "Soft Focus",
                           "Crisp Focus", "Selective Focus", "Zone Focus", "Dreamy Bokeh", "Swirly Bokeh", "Creamy Bokeh"],
        "composition_rule": ["Rule of Thirds", "Golden Ratio", "Center Composition", "Symmetrical",
                             "Asymmetrical", "Leading Lines", "Framing", "Depth Layering",
                             "Foreground Focus", "Background Focus", "Negative Space", "Filling Frame",
                             "Diagonal Composition", "Triangular Composition", "Circular Composition",
                             "Rule of Odds", "Golden Triangle", "Dynamic Symmetry", "Pattern Breaking", "Visual Weight"],
        "lighting_quality": ["Hard Light", "Soft Light", "Diffused Light", "Direct Light", "Bounced Light",
                             "Filtered Light", "Natural Light", "Artificial Light", "Mixed Lighting",
                             "High Contrast", "Low Contrast", "Even Lighting", "Dramatic Lighting",
                             "Subtle Lighting", "Harsh Lighting", "Gentle Lighting", "Softbox Lighting", "Ring Light"],
        "lighting_setup": ["Three-Point Lighting", "Key Light Only", "Rembrandt Lighting", "Butterfly Lighting",
                           "Split Lighting", "Loop Lighting", "Broad Lighting", "Short Lighting",
                           "Rim Lighting", "Backlighting", "Side Lighting", "Top Lighting", "Bottom Lighting",
                           "Practical Lighting", "Motivated Lighting", "Available Light", "High Key Setup", "Low Key Setup"],
        "lighting_color": ["Warm 3200K", "Neutral 5600K", "Cool 7000K", "Daylight 5500K", "Tungsten 3200K",
                           "LED 4000K", "Fluorescent 4100K", "Candlelight 1900K", "Sunset 2000K",
                           "Blue Hour 15000K", "Overcast 6500K", "Golden Hour 2500K",
                           "Neon Colors", "Colored Gels", "Practical Colors", "Red Light", "Blue Light", "Green Light", "Purple Light"],
        "character_gender": ["Male", "Female", "Androgynous", "Non-binary", "Genderfluid", "Transgender Male", "Transgender Female"],
        "character_age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly", "Infant", "Senior Citizen", "Adolescent", "Adult"],
        "character_ethnicity": ["Caucasian", "Asian", "African", "Hispanic", "Middle Eastern",
                                "South Asian", "Mixed", "East Asian", "Southeast Asian", "Native American", "Pacific Islander",
                                "Latin American", "Indigenous", "African American", "European", "Oceanic"],
        "character_clothing": ["Casual", "Formal", "Fantasy Armor", "Sci-Fi Suit", "Historical Attire",
                               "Modern Fashion", "Minimalist", "Elaborate", "Steampunk Outfit", "Cyberpunk Gear", "Tattered Rags", "Elegant Gown",
                               "Sportswear", "Business Attire", "Military Uniform", "Traditional Dress", "Futuristic Robes", "Punk Rock Style"],
        "character_expression": ["Happy", "Sad", "Angry", "Surprised", "Fearful", "Confused",
                                 "Determined", "Calm", "Intense", "Playful", "Seductive", "Melancholy", "Furious", "Joyful",
                                 "Thoughtful", "Mischievous", "Exhausted", "Excited", "Disgusted", "Annoyed", "Hopeful"],
        "visual_style": ["Photorealistic", "Stylized", "Artistic", "Abstract", "Surreal", "Impressionistic",
                         "Expressionistic", "Minimalist", "Maximalist", "Geometric", "Organic",
                         "Textural", "Smooth", "Rough", "Polished", "Raw", "Refined", "Gritty", "Hand-drawn", "Digital Art"],
        "image_quality": ["Standard Definition", "High Definition", "Ultra HD 4K", "8K Resolution",
                          "Cinematic 4K", "IMAX Quality", "Web Optimized", "Print Quality",
                          "Broadcast Quality", "Cinema Quality", "Ultra High Quality",
                          "Maximum Detail", "Sharp", "Soft", "Crisp", "Professional Grade", "Lossless", "High Fidelity"],
        "environment_type": ["Interior", "Exterior", "Studio", "Location", "Practical Location",
                             "Urban", "Suburban", "Rural", "Wilderness", "Industrial", "Residential",
                             "Commercial", "Institutional", "Historical", "Modern", "Futuristic",
                             "Underwater", "Space", "Desert", "Forest", "Mountain", "Arctic", "Tropical", "Volcanic", "Canyon"],
        "atmospheric_effects": ["Volumetric Light", "God Rays", "Lens Flare", "Light Shafts", "Dust Particles",
                                "Smoke", "Steam", "Mist", "Fog Bank", "Heat Shimmer", "Light Pollution",
                                "Atmospheric Haze", "Particle Effects", "Floating Debris", "Pollen", "Ash",
                                "Rain Streaks", "Snow Flurries", "Aurora Borealis", "Lightning Strikes",
                                "Sandstorm", "Blizzard", "Heavy Rain", "Drizzle", "Gloom", "Ethereal Mist"],
        "time_of_day": ["Pre-Dawn", "Dawn", "Early Morning", "Mid Morning", "Late Morning",
                        "Noon", "Early Afternoon", "Mid Afternoon", "Late Afternoon",
                        "Dusk", "Twilight", "Blue Hour", "Golden Hour", "Night",
                        "Late Night", "Midnight", "Pre-Dawn Hours", "Sunrise", "Sunset"],
        "weather_atmosphere": ["Clear", "Partly Cloudy", "Overcast", "Stormy", "Rainy", "Drizzling",
                               "Heavy Rain", "Thunderstorm", "Snowing", "Light Snow", "Heavy Snow", "Blizzard",
                               "Foggy", "Misty", "Hazy", "Dusty", "Windy", "Calm", "Humid", "Dry", "Sunny", "Cloudy"],
        "season": ["Spring", "Summer", "Autumn", "Winter",
                   "Early Spring", "Late Spring", "Mid Summer", "Late Summer",
                   "Early Fall", "Late Fall", "Early Winter", "Late Winter", "Monsoon", "Dry Season", "Wet Season"],
        "post_processing": ["HDR", "Tone Mapping", "Color Correction", "Color Grading", "Contrast Enhancement",
                            "Saturation Boost", "Desaturation", "Sharpening", "Noise Reduction",
                            "Film Emulation", "Digital Enhancement", "Vintage Processing", "Modern Processing",
                            "Artistic Filter", "Realistic Processing", "Stylized Processing", "Chromatic Aberration Correction", "Vignette Addition"],
        "lens_effects": ["Lens Flare", "Chromatic Aberration", "Vignetting", "Barrel Distortion",
                         "Pincushion Distortion", "Bokeh Highlights", "Lens Ghosting", "Glare",
                         "Diffraction Spikes", "Anamorphic Flares", "Vintage Lens Character",
                         "Clean Lens", "Dirty Lens", "Water Drops", "Lens Breathing", "Star Filter", "Mist Filter"],
        "motion_blur": ["Subtle Motion Blur", "Moderate Motion Blur", "Heavy Motion Blur",
                        "Directional Motion Blur", "Radial Motion Blur", "No Motion Blur", "Zoom Blur", "Spin Blur"],
        "cinematic_technique": ["Montage", "Long Take", "Quick Cuts", "Match Cut", "Jump Cut", "Cross Cut",
                                "Fade In", "Fade Out", "Dissolve", "Wipe", "Iris", "Split Screen",
                                "Freeze Frame", "Slow Motion", "Time Lapse", "Reverse Motion",
                                "Seamless Transition", "Hard Cut", "Smash Cut", "L-Cut", "J-Cut", "Flashback", "Flashforward"],
        "genre_influence": ["Film Noir", "Science Fiction", "Fantasy", "Horror", "Thriller", "Drama",
                            "Comedy", "Romance", "Action", "Adventure", "Western", "Crime",
                            "Mystery", "Documentary", "Art House", "Experimental", "Period Piece",
                            "Biopic", "Musical", "War Film", "Disaster", "Superhero", "Spy", "Legal", "Medical", "Sports"],
        "mood_emotion": ["Joyful", "Melancholic", "Mysterious", "Dramatic", "Peaceful", "Tense",
                         "Romantic", "Threatening", "Whimsical", "Serious", "Playful", "Somber",
                         "Energetic", "Calm", "Chaotic", "Ordered", "Intimate", "Epic",
                         "Nostalgic", "Futuristic", "Timeless", "Urgent", "Contemplative", "Anxious", "Serene", "Exciting", "Depressing"],
        "animation_style": ["Cel Animation", "CGI Animation", "Stop Motion", "Rotoscoping",
                            "2D Animation", "3D Animation", "Anime Style", "Cartoon Style",
                            "Motion Graphics", "Puppet Animation", "Claymation", "Pixel Art Animation",
                            "Hand-drawn Animation", "Vector Animation", "Stop-motion Clay", "Paper Cut-out",
                            "Disney Animation", "Pixar 3D Animation", "Warner Bros. Classic Animation",
                            "80s Anime Style", "Ghibli Studio Style", "Ukiyo-e Animation",
                            "Sand Animation", "Pin-screen Animation", "Oil Paint Animation",
                            "Charcoal Animation", "Silhouette Animation", "Experimental Animation", "Abstract Animation"],
        # Added film_emulation to ADJUSTABLE_DETAIL_OPTIONS
        "film_emulation": ["35mm Film Grain", "16mm Film Grain", "Super 8 Film Grain", "Digital Clean",
                           "Vintage Film Look", "Modern Film Look", "Kodachrome", "Technicolor",
                           "Black and White Film", "Sepia Tone Film", "Cross Processed Film",
                           "Arri Alexa Look", "Red Camera Look", "Sony Venice Look", "Film Stock Emulation"],
    }

    # Categorized list of adjustable details for INPUT_TYPES and get_input_property
    ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED = {
        "camera_controls": [
            ('shot_type', 'shot_type_weight', "Type of shot (e.g., Close-Up, Wide Shot)."),
            ('camera_angle', 'camera_angle_weight', "Camera angle (e.g., Low Angle, Bird's Eye View)."),
            ('camera_movement', 'camera_movement_weight', "Camera movement (e.g., Dolly In, Handheld)."),
            ('camera_lens', 'camera_lens_weight', "Type of camera lens (e.g., Wide Angle 24mm, Anamorphic)."),
            ('aspect_ratio', 'aspect_ratio_weight', "Output aspect ratio (e.g., 16:9 Widescreen, 2.35:1 Anamorphic)."),
            ('depth_of_field', 'depth_of_field_weight', "Depth of field effect (e.g., Shallow DOF, Bokeh Heavy)."),
            ('composition_rule', 'composition_rule_weight', "Compositional rule applied (e.g., Rule of Thirds, Leading Lines, Golden Ratio)."),
        ],
        "lighting_controls": [
            ('lighting_quality', 'lighting_quality_weight', "Quality of light (e.g., Hard Light, Soft Light, Dramatic Lighting)."),
            ('lighting_setup', 'lighting_setup_weight', "Lighting setup (e.g., Three-Point Lighting, Backlighting)."),
            ('lighting_color', 'lighting_color_weight', "Color temperature or specific color of light (e.g., Golden Hour, Neon Colors)."),
        ],
        "character_controls": [
            ('character_gender', 'character_gender_weight', "Gender representation of the character."),
            ('character_age', 'character_age_weight', "Age representation of the character."),
            ('character_ethnicity', 'character_ethnicity_weight', "Ethnicity representation of the character."),
            ('character_clothing', 'character_clothing_weight', "Clothing style of the character."),
            ('character_expression', 'character_expression_weight', "Facial expression of the character."),
        ],
        "visual_style_controls": [
            ('visual_style', 'visual_style_weight', "Overall visual style of the image (e.g., Photorealistic, Artistic, Gritty)." ),
            ('image_quality', 'image_quality_weight', "Overall image resolution or quality (e.g., 8K Resolution, Cinematic 4K, Sharp)."),
        ],
        "environment_controls": [
            ('environment_type', 'environment_type_weight', "Type of environment (e.g., Urban, Natural Landscape, Interior)."),
            ('atmospheric_effects', 'atmospheric_effects_weight', "Specific atmospheric visual effects (e.g., God Rays, Lens Flare, Smoke)."),
        ],
        "time_weather_controls": [
            ('time_of_day', 'time_of_day_weight', "Time of day for the scene (e.g., Golden Hour, Night)."),
            ('weather_atmosphere', 'weather_atmosphere_weight', "Weather or general atmospheric condition (e.g., Foggy, Rainy, Clear)."),
            ('season', 'season_weight', "Season of the year (e.g., Autumn, Winter)."),
        ],
        "post_vfx_controls": [
            ('post_processing', 'post_processing_weight', "Post-production effects applied (e.g., HDR, Tone Mapping, Sharpening)."),
            ('lens_effects', 'lens_effects_weight', "Specific lens artifacts or effects (e.g., Lens Flare, Chromatic Aberration, Bokeh Highlights)."),
            ('motion_blur', 'motion_blur_weight', "Amount and type of motion blur."),
            ('film_emulation', 'film_emulation_weight', "Emulation of film stock or digital camera (e.g., 35mm Film, Arri Alexa, Film Grain Heavy)."),
            ('cinematic_technique', 'cinematic_technique_weight', "Specific cinematic technique (e.g., Long Take, Slow Motion, Montage)."),
        ],
        "misc_controls": [
            ('genre_influence', 'genre_influence_weight', "Influence from a film or art genre (e.g., Film Noir, Sci-Fi, Horror)."),
            ('mood_emotion', 'mood_emotion_weight', "Emotional tone or mood of the scene (e.g., Dramatic, Peaceful, Mysterious)."),
            ('animation_style', 'animation_style_weight', "Specific animation style (e.g., Cel Animation, CGI Animation)."),
        ]
    }

    def __init__(self):
        print("DEBUG: CinematicSceneDirectorTools class instantiated.")
        self._last_llm_generated_prompt = "" 
        self._last_negative_prompt = "" 

        self.quality_words = {
            "none": ["enhanced", "detailed", "creative", "expanded", "richly described"],
            "photography": ["professional photography", "high resolution", "detailed", "sharp focus", "realistic", "crisp"],
            "artistic": ["artistic masterpiece", "beautiful", "creative", "expressive", "painterly", "stylized"],
            "cinematic": ["cinematic lighting", "dramatic", "epic", "movie scene", "widescreen", "film look"],
            "realistic": ["photorealistic", "ultra detailed", "lifelike", "high quality", "authentic", "natural"],
            "animation": ["fluid animation", "dynamic motion", "expressive characters", "vibrant colors", "smooth transitions"],
            "concept art": ["imaginative concept art", "illustrative", "visionary", "detailed design", "world-building"],
            "stylized": ["unique style", "distinctive aesthetic", "graphic novel style", "comic book art", "abstract elements"],
            "abstract": ["abstract forms", "non-representational", "conceptual", "surreal", "impressionistic"],
            "documentary": ["authentic footage", "raw realism, observational", "unfiltered", "gritty"]
        }
        self.enhancement_templates = {
            "none": "Expand and enhance this prompt with descriptive details: {prompt}",
            "photography": "Enhance this photography prompt with professional details: {prompt}",
            "artistic": "Create an artistic enhancement for: {prompt}",
            "cinematic": "Add cinematic elements to: {prompt}",
            "realistic": "Make this more realistic and detailed: {prompt}",
            "animation": "Describe this as a scene from a high-quality animation, focusing on motion and style: {prompt}",
            "concept art": "Generate a concept art description based on: {prompt}",
            "stylized": "Re-imagine this prompt with a distinct stylized aesthetic: {prompt}",
            "abstract": "Transform this into an abstract interpretation of: {prompt}",
            "documentary": "Describe this scene as if it were from an authentic documentary, focusing on realism: {prompt}"
        }

        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.model_mappings = {
            "Qwen2.5-1.5B [Best Quality]": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B [High Quality]": "Qwen/Qwen2.5-3B-Instruct", 
            "Qwen2.5-7B [Maximum Quality]": "Qwen/Qwen2.5-7B-Instruct",
            "Llama-3.2-1B [Fast]": "meta-llama/Llama-3.2-1B-Instruct",
            "Llama-3.2-3B [Balanced]": "meta-llama/Llama-3.2-3B-Instruct",
            "Florence-2-base-ft": "microsoft/Florence-2-base-ft" 
        }

    @classmethod
    def get_normalized_options(cls):
        """
        Returns the original, unpadded options lists.
        The 'normalization' logic (padding with None) is removed
        as it's not needed for ComfyUI dropdowns and causes visual clutter.
        """
        # Return original presets directly
        normalized_presets = dict(cls.PRESETS)

        # Return original negative prompt presets directly
        normalized_negative_presets = dict(cls.NEGATIVE_PROMPT_PRESETS)

        # Return original adjustable details directly
        normalized_adjustable_details = dict(cls.ADJUSTABLE_DETAIL_OPTIONS)
            
        return normalized_presets, normalized_negative_presets, normalized_adjustable_details

    @classmethod
    def INPUT_TYPES(cls):
        normalized_presets, normalized_negative_presets, normalized_adjustable_details = cls.get_normalized_options()

        # Dynamically generate preset dropdowns
        preset_inputs = {}
        for preset_name, preset_dict in normalized_presets.items():
            display_name = " ".join(word.capitalize() for word in preset_name.split('_'))
            # Use keys directly for display list, no "None_X" padding
            display_keys = list(preset_dict.keys())
            preset_inputs[preset_name] = (["None"] + display_keys, {"default": "None", "tooltip": f"Apply a {display_name.lower()} preset."})

        # Dynamically generate adjustable detail inputs
        adjustable_detail_inputs = {}
        for category_params in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED.values():
            for param_name, weight_name, tooltip in category_params:
                # Defensive check: Ensure param_name exists in normalized_adjustable_details
                values_list = ["None"]
                if param_name in normalized_adjustable_details:
                    # Use original options list, no "None" padding
                    values_list += normalized_adjustable_details[param_name]
                else:
                    # Fallback for unexpected missing keys (should not happen if definitions are consistent)
                    print(f"WARNING: '{param_name}' not found in normalized_adjustable_details. Using fallback values.")
                    values_list += ["Default Option 1", "Default Option 2"] # Provide some dummy values

                adjustable_detail_inputs[param_name] = (values_list, { # Changed to use values_list directly for dropdown
                    "default": "None",
                    "tooltip": tooltip
                })
                adjustable_detail_inputs[weight_name] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1, "tooltip": f"Weight for the {param_name.replace('_', ' ')}."})


        return {
            "required": {
                "master_prompts": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter multiple prompts separated by line breaks. E.g., 'A dramatic scene\\nA mysterious figure'.",
                    "tooltip": "Enter multiple prompts separated by line breaks. The node will cycle through them automatically. This list is used as the base for LLM enhancement if LLM inference is enabled."
                }),
                "prompt_index": ("INT", { # Renamed to prompt_index and made required
                    "default": 0, 
                    "min": 0, # Changed min to 0 as random selection is removed from here
                    "max": 999, 
                    "step": 1,
                    "tooltip": "Current prompt index. This index will be used to select a prompt from the master prompts list."
                }),
            },
            "optional": {
                # LLM Positive Prompt Enhancement
                "enable_llm_inference": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, a local LLM will be used to enhance or generate the core prompt based on the master prompts list."
                }),
                "llm_model_selection": ([
                    "Qwen2.5-1.5B [Best Quality]",
                    "Qwen2.5-3B [High Quality]", 
                    "Qwen2.5-7B [Maximum Quality]","Llama-3.2-1B [Fast]",
                    "Llama-3.2-3B [Balanced]",
                    "Florence-2-base-ft", 
                    "Fallback [No Model]"
                ], {"default": "Qwen2.5-1.5B [Best Quality]", "tooltip": "Select a local HuggingFace LLM for prompt enhancement. Smaller models (e.g., 1.5B, 1B) use less VRAM. 'Fallback' uses rule-based enhancement. Requires 'transformers' library."}),
                "llm_enhancement_style": ([
                    "none", 
                    "photography",
                    "artistic", 
                    "cinematic",
                    "realistic",
                    "animation",
                    "concept art",
                    "stylized",
                    "abstract",
                    "documentary"
                ], {"default": "none", "tooltip": "Style preference for LLM enhancement. 'None' provides a general enhancement."}),
                "llm_creativity_level": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Controls the randomness and creativity of the LLM output (higher = more creative)."
                }),
                "llm_max_length": ("INT", {
                    "default": 150, # Adjusted default max length to 150
                    "min": 50,
                    "max": 500,
                    "step": 5,
                    "display": "slider",
                    "tooltip": "Maximum length of the LLM-generated enhancement."
                }),
                "llm_purge_cache": ("BOOLEAN", {
                    "default": False, # Default is False, meaning model stays in VRAM.
                    "tooltip": "If checked (True), unloads the LLM model from VRAM after each generation to free up memory. This can cause slower subsequent generations due to reloading. If unchecked (False), LLM stays in VRAM, consuming more memory but potentially faster for repeated use."
                }),
                "llm_regenerate_on_each_run": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If checked, LLM will generate a new prompt each run. If unchecked, it will reuse the last generated LLM prompt."
                }),
                "incorporate_adjustments_into_llm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If LLM inference is enabled, incorporate selected presets and individual adjustments into the LLM's prompt for richer generation."
                }),
                "enable_preset_override_button": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, only the first selected preset will be used, ignoring all other presets and individual adjustments."
                }),

                # Negative Prompt Controls (Simplified)
                "negative_prompt_preset": (list(normalized_negative_presets.keys()), {
                    "default": "None",
                    "tooltip": "Select a preset for the negative prompt. Choosing 'None' will result in an empty negative prompt."
                }),
                
                # Organized Presets
                "--- Presets ---": ("BOOLEAN", {"default": True, "hidden": True}), # Separator
                **{k: preset_inputs[k] for k in ["cinematic_preset", "genre_preset", "visual_style_preset", "lighting_preset", "environment_preset", "character_preset", "vfx_preset", "humor_preset", "horror_preset", "nsfw_preset", "hentai_preset", "domain_specific_preset"]},
                
                # Organized Adjustable Details
                "--- Adjustable Details ---": ("BOOLEAN", {"default": True, "hidden": True}), # Separator
                "--- Camera Controls ---": ("BOOLEAN", {"default": True, "hidden": True}),
                **{k: adjustable_detail_inputs[k] for k in [p[0] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["camera_controls"]] + [p[1] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["camera_controls"]]},
                
                "--- Lighting Controls ---": ("BOOLEAN", {"default": True, "hidden": True}),
                **{k: adjustable_detail_inputs[k] for k in [p[0] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["lighting_controls"]] + [p[1] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["lighting_controls"]]},
                
                "--- Character Controls ---": ("BOOLEAN", {"default": True, "hidden": True}),
                **{k: adjustable_detail_inputs[k] for k in [p[0] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["character_controls"]] + [p[1] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["character_controls"]]},
                
                "--- Visual Style Controls ---": ("BOOLEAN", {"default": True, "hidden": True}),
                **{k: adjustable_detail_inputs[k] for k in [p[0] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["visual_style_controls"]] + [p[1] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["visual_style_controls"]]},
                
                "--- Environment Controls ---": ("BOOLEAN", {"default": True, "hidden": True}),
                **{k: adjustable_detail_inputs[k] for k in [p[0] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["environment_controls"]] + [p[1] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["environment_controls"]]},
                
                "--- Time & Weather Controls ---": ("BOOLEAN", {"default": True, "hidden": True}),
                **{k: adjustable_detail_inputs[k] for k in [p[0] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["time_weather_controls"]] + [p[1] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["time_weather_controls"]]},
                
                "--- Post & VFX Controls ---": ("BOOLEAN", {"default": True, "hidden": True}),
                **{k: adjustable_detail_inputs[k] for k in [p[0] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["post_vfx_controls"]] + [p[1] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["post_vfx_controls"]]},
                
                "--- Miscellaneous Controls ---": ("BOOLEAN", {"default": True, "hidden": True}),
                **{k: adjustable_detail_inputs[k] for k in [p[0] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["misc_controls"]] + [p[1] for p in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED["misc_controls"]]},
            }
        }

    RETURN_TYPES = ("STRING", "STRING") # Now returns two strings
    RETURN_NAMES = ("cinematic_prompt", "negative_prompt") # Renamed for clarity
    FUNCTION = "generate_prompt_string"
    CATEGORY = "BH Tools" 
    OUTPUT_NODE = True # <--- Added this line to make it an output node

    @classmethod
    def get_input_property(cls, values, property_name):
        # Determine current modes
        enable_llm = values.get("enable_llm_inference", False)
        enable_preset_override = values.get("enable_preset_override_button", False)
        incorporate_adjustments = values.get("incorporate_adjustments_into_llm", False)

        # Prompt Indexing Controls visibility (only prompt_index remains, always visible)
        if property_name == "prompt_index":
            return {"hidden": False}

        # LLM Positive Prompt specific inputs
        llm_positive_input_names = [
            "llm_model_selection", "llm_enhancement_style", 
            "llm_creativity_level", "llm_max_length", "llm_purge_cache",
            "llm_regenerate_on_each_run", "incorporate_adjustments_into_llm" 
        ]
        if property_name in llm_positive_input_names:
            return {"hidden": not enable_llm} 

        # Negative Prompt Preset is always visible
        if property_name == "negative_prompt_preset":
            return {"hidden": False}
        
        # Core prompt inputs (master_prompts) are always visible
        if property_name == "master_prompts":
            return {} # Always visible
            
        # Preset dropdowns visibility
        preset_dropdowns = list(cls.PRESETS.keys())
        if property_name in preset_dropdowns:
            # Hidden if LLM is enabled AND adjustments are NOT incorporated, OR if preset override is active
            return {"hidden": (enable_llm and not incorporate_adjustments) or enable_preset_override}
        
        # Individual adjustable details visibility
        # Hidden if LLM is enabled AND adjustments are NOT incorporated, OR if preset override is active
        for category_params in cls.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED.values():
            for param_name, weight_name, _ in category_params:
                if property_name == param_name or property_name == weight_name:
                    return {"hidden": (enable_llm and not incorporate_adjustments) or enable_preset_override}
        
        # enable_preset_override_button is always visible
        if property_name == "enable_preset_override_button":
            return {}

        # Handle visibility for separators
        if property_name in ["--- Presets ---", "--- Adjustable Details ---", "--- Camera Controls ---",
                              "--- Lighting Controls ---", "--- Character Controls ---", "--- Visual Style Controls ---",
                              "--- Environment Controls ---", "--- Time & Weather Controls ---",
                              "--- Post & VFX Controls ---", "--- Miscellaneous Controls ---"]:
            # Separators for presets are hidden if presets are hidden
            if property_name == "--- Presets ---":
                return {"hidden": (enable_llm and not incorporate_adjustments) or enable_preset_override}
            # Separators for adjustable details are hidden if adjustable details are hidden
            elif property_name.startswith("---") and property_name.endswith("Controls ---"):
                return {"hidden": (enable_llm and not incorporate_adjustments) or enable_preset_override}
            return {"hidden": False} # Default for other separators

        return {} # Default to visible if not explicitly hidden

    def parse_master_prompts(self, master_prompts_text):
        """Parse the master prompts from multiline text, filtering out empty lines."""
        if not master_prompts_text or not master_prompts_text.strip():
            return ["A cinematic scene."]
        
        prompts = [line.strip() for line in master_prompts_text.split('\n') if line.strip()]
        
        if not prompts:
            return ["A cinematic scene."]
            
        return prompts

    def get_current_prompt_index(self, master_prompts_list, prompt_index):
        """
        Determines the effective prompt index based on the provided manual index.
        """
        total_prompts = len(master_prompts_list)
        print(f"DEBUG: Total prompts in list: {total_prompts}") 
        if total_prompts == 0:
            print("DEBUG: Master prompts list is empty, returning index 0.")
            return 0 

        # Ensure the prompt_index is within the valid range
        effective_index = prompt_index % total_prompts
        print(f"DEBUG: Manual Index - Effective index: {effective_index}")
        return effective_index

    def get_current_prompt_text(self, master_prompts_list, effective_index): 
        """Get the current prompt text based on the effective index."""
        if not master_prompts_list:
            return "A cinematic scene."
        return master_prompts_list[effective_index]


    def weighted_format(self, value, weight):
        """Format a value with its weight for prompt inclusion."""
        # Ensure value is a string and not 'None' or empty after stripping
        if not isinstance(value, str) or value.strip().lower() == "none" or not value.strip():
            return None
        
        value = value.strip().lower()
        
        # Ensure weight is a float, default to 1.0 if conversion fails
        try:
            rounded_weight = round(float(weight), 1)
        except (ValueError, TypeError):
            rounded_weight = 1.0 
            
        if rounded_weight == 1.0:
            return value
        else:
            return f"({value}:{rounded_weight})"

    def format_setting(self, value, weight, prefix="", suffix=""):
        """Format a setting with optional prefix and suffix."""
        formatted = self.weighted_format(value, weight)
        if formatted is None:
            return None
        return f"{prefix}{formatted}{suffix}"

    def get_model_name(self, model_selection):
        """Maps user-friendly model selection to HuggingFace model name."""
        return self.model_mappings.get(model_selection, None)

    def load_model_optimized(self, model_name):
        """
        Loads the specified HuggingFace model and tokenizer with optimizations.
        Checks if the model is already loaded to avoid redundant loading.
        """
        if not TRANSFORMERS_AVAILABLE:
            print(f"\n--- WARNING: 'transformers' library not found. Cannot load LLM '{model_name}'. ---")
            print("--- Please install it with: pip install transformers torch accelerate bitsandbytes sentencepiece ---")
            return False
            
        if self.current_model_name == model_name and self.model is not None:
            # Model is already loaded
            return True
            
        # Clean up any previously loaded model before loading a new one
        self.cleanup_model()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            quantization_config = None
            if BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
                # Use 4-bit quantization for VRAM efficiency on CUDA
                print(f"DEBUG: bitsandbytes available, attempting 4-bit quantization for {model_name}")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, # Use float16 for compute for 4090
                    bnb_4bit_use_double_quant=True,
                )
            elif torch.cuda.is_available():
                print(f"\n--- WARNING: bitsandbytes not found. LLM '{model_name}' will load in full precision (float16/float32) and use more VRAM. ---")
                print("--- Consider installing it with: pip install bitsandbytes ---")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, # Load in float16 on GPU for VRAM efficiency if no 4bit
                device_map="auto" if torch.cuda.is_available() else None, # Automatically map layers to available devices
                trust_remote_code=True,
                low_cpu_mem_usage=True, # Helps reduce CPU RAM during loading
                quantization_config=quantization_config # Apply quantization if configured
            )
            
            self.current_model_name = model_name
            print(f"DEBUG: Successfully loaded LLM model: {model_name}")
            return True
            
        except Exception as e:
            print(f"\n--- ERROR loading LLM model {model_name}: {str(e)} ---")
            print("--- This could be due to missing libraries, insufficient VRAM, or network issues. ---")
            print("--- Ensure 'transformers', 'torch', 'accelerate', and 'bitsandbytes' are installed and compatible. ---")
            print("--- Falling back to rule-based enhancement. ---\n")
            self.cleanup_model() # Ensure model is cleaned up on failure
            return False
            
    def cleanup_model(self):
        """
        Unloads the current model and tokenizer from memory and clears CUDA cache.
        This helps free up VRAM and RAM.
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_name = None
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 

    def generate_with_model(self, prompt, style, creativity_level, max_length, prompt_type="positive"):
        """
        Generates an enhanced prompt using the loaded HuggingFace model.
        """
        try:
            if prompt_type == "negative":
                template = "Enhance this negative prompt with more descriptive undesirable elements: {prompt}"
                system_message = (
                    f"You are a negative prompt engineer. Expand this negative prompt with visual undesirable elements. "
                    f"Focus purely on concrete, visual elements that should not be in an image. "
                    f"Avoid metaphors, abstract concepts, or any language that does not translate directly into visual information. "
                    f"Keep the response concise, ideally within {max_length} tokens."
                )
            else: # positive
                template = self.enhancement_templates.get(style, self.enhancement_templates["none"])
                # Modified system message for positive prompt
                system_message = (
                    f"You are a prompt engineer. Enhance this prompt with descriptive visual details only. "
                    f"Describe only what can be seen in an image. Avoid metaphors, thoughts, or abstract concepts. "
                    f"Keep the response concise, ideally within {max_length} tokens."
                )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": template.format(prompt=prompt)}
            ]
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback for models without chat template (shouldn't happen with Qwen/Llama-3)
                formatted_prompt = f"{system_message}\nUser: {template.format(prompt=prompt)}"
            
            # Adjusted max_length for tokenizer input to prevent errors
            # The value 60 is chosen to be safely below the reported model limit of 77 tokens.
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True, 
                max_length=self.tokenizer.model_max_length if self.tokenizer.model_max_length > 0 else 512, # Use model's max length or a safe default
                padding=True
            )
            
            if torch.cuda.is_available() and self.model.device.type == 'cuda':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad(): 
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length, 
                    temperature=creativity_level, 
                    do_sample=True, 
                    pad_token_id=self.tokenizer.pad_token_id, 
                    eos_token_id=self.tokenizer.eos_token_id, 
                    repetition_penalty=1.1,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            enhanced = self.extract_enhancement(generated_text, formatted_prompt)
            return self.clean_output(enhanced)
            
        except Exception as e:
            print(f"\n--- ERROR during LLM generation: {str(e)} ---")
            print("--- Falling back to rule-based enhancement. ---\n")
            return self._create_fallback_enhancement(prompt, style, creativity_level, prompt_type)

    def extract_enhancement(self, generated_text, instruction):
        """
        Extracts the generated enhancement from the full LLM output.
        This handles cases where the LLM might echo the input or system messages.
        """
        # Attempt to find the "assistant" turn and take everything after it
        assistant_marker = "assistant\n"
        if assistant_marker in generated_text.lower():
            parts = generated_text.lower().split(assistant_marker, 1)
            enhanced = parts[1].strip() if len(parts) > 1 else generated_text
        else:
            # Fallback: remove the original instruction if "assistant" marker isn't found
            enhanced = generated_text.replace(instruction, "").strip()
        
        # Remove common introductory phrases from the start of the generated text
        intro_patterns = [
            r'^(Okay, here is the enhanced prompt:|Here is your enhanced prompt:|Enhanced prompt:)',
            r'^(Here\'s an enhanced version of your prompt:|Here\'s the enhanced prompt:)',
            r'^(A|An)\s+', # Remove leading "A " or "An " if it's the start of the generated text
        ]
        for pattern in intro_patterns:
            enhanced = re.sub(pattern, '', enhanced, flags=re.IGNORECASE).strip()
        
        return enhanced

    def clean_output(self, text):
        """
        Cleans and formats the final output string, removing unwanted patterns
        and ensuring proper spacing.
        """
        # Remove any remaining chat template artifacts (e.g., <|im_start|>user, <|im_end|>)
        text = re.sub(r'<\|im_start\|>\w+\n?', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)
        
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing punctuation if it's not part of the sentence structure
        text = re.sub(r'^[.,!?;:]+', '', text).strip()
        text = re.sub(r'[.,!?;:]+$', '', text).strip() # Remove trailing punctuation
        
        # Ensure it doesn't start with a comma or other separator
        if text and text[0] in ',;':
            text = text[1:].strip()
            
        # Capitalize the first letter if it's a letter
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
            
        return text

    def _create_fallback_enhancement(self, prompt, style, creativity_level, prompt_type="positive"):
        """Create a fallback enhancement when LLM is not available or fails."""
        if prompt_type == "negative":
            # For negative prompts, a simpler fallback
            return f"{prompt}, blurry, distorted, ugly, deformed, bad anatomy, watermark, text".strip()

        quality_words = self.quality_words.get(style, self.quality_words["none"])
        
        num_words = min(3, max(1, int(creativity_level * 4)) + 1) 
        selected_words = random.sample(quality_words, min(num_words, len(quality_words)))
        
        enhancement = ", ".join(selected_words)
        
        if creativity_level > 0.7:
            extra_details = {
                "none": "highly detailed, complex",
                "photography": "8K resolution, professional lighting",
                "artistic": "masterpiece, award winning",
                "cinematic": "dramatic composition, epic scale",
                "realistic": "hyperrealistic, ultra detailed",
                "animation": "smooth motion, expressive characters",
                "concept art": "detailed concept, imaginative design",
                "stylized": "unique visual style, artistic flair",
                "abstract": "complex patterns, symbolic imagery",
                "documentary": "raw, unedited, authentic feel"
            }
            enhancement += f", {extra_details.get(style, 'high quality')}"
        
        return f"{prompt}, {enhancement}".strip()

    def generate_negative_prompt(self, negative_prompt_preset_name):
        """
        Generates the negative prompt based on preset. LLM enhancement for negative prompts is removed.
        """
        selected_negative_prompt_text = self.NEGATIVE_PROMPT_PRESETS.get(negative_prompt_preset_name, "")
        return selected_negative_prompt_text


    def generate_prompt_string(self, **kwargs):
        """
        Main function to generate the cinematic prompt string.
        Combines master prompts with comprehensive cinematic parameters,
        optionally enhanced by a local LLM.
        """
        
        # Extract parameters for positive prompt
        master_prompts = kwargs.get('master_prompts', '')
        prompt_index = kwargs.get('prompt_index', 0) # Use the single prompt_index

        enable_llm_inference = kwargs.get('enable_llm_inference', False)
        llm_model_selection = kwargs.get('llm_model_selection', 'Qwen2.5-1.5B [Best Quality]') 
        llm_enhancement_style = kwargs.get('llm_enhancement_style', 'none')
        llm_creativity_level = kwargs.get('llm_creativity_level', 0.7)
        llm_max_length = kwargs.get('llm_max_length', 150) # Updated default
        llm_purge_cache = kwargs.get('llm_purge_cache', False)
        llm_regenerate_on_each_run = kwargs.get('llm_regenerate_on_each_run', True)
        incorporate_adjustments_into_llm = kwargs.get('incorporate_adjustments_into_llm', False)
        enable_preset_override_button = kwargs.get('enable_preset_override_button', False)

        # Extract parameters for negative prompt
        negative_prompt_preset_name = kwargs.get('negative_prompt_preset', 'None')


        # --- Positive Prompt Generation ---
        master_prompts_list = self.parse_master_prompts(master_prompts)
        
        # Determine the effective prompt index using the simplified method
        effective_prompt_index = self.get_current_prompt_index(
            master_prompts_list, prompt_index
        )
        base_text_for_llm = self.get_current_prompt_text(master_prompts_list, effective_prompt_index) 
        print(f"DEBUG: Selected base prompt: '{base_text_for_llm}' (from index {effective_prompt_index})") # New debug print

        # Defensive checks for LLM model and style
        valid_llm_models = self.INPUT_TYPES()["optional"]["llm_model_selection"][0]
        if llm_model_selection not in valid_llm_models:
            llm_model_selection = 'Fallback [No Model]'
        valid_llm_styles = self.INPUT_TYPES()["optional"]["llm_enhancement_style"][0]
        if llm_enhancement_style not in valid_llm_styles:
            llm_enhancement_style = 'none'

        current_master_prompt = ""
        if enable_llm_inference:
            llm_additional_context = ""
            if incorporate_adjustments_into_llm:
                temp_components_for_llm = [] # This list will hold components specifically for LLM input
                normalized_presets, _, normalized_adjustable_details = self.get_normalized_options()

                if enable_preset_override_button:
                    preset_category_order = list(normalized_presets.keys())
                    for category_name in preset_category_order:
                        selected_preset_name = kwargs.get(category_name, 'None')
                        if selected_preset_name != "None" and selected_preset_name in normalized_presets[category_name]:
                            preset_text = normalized_presets[category_name][selected_preset_name]
                            temp_components_for_llm.append(preset_text)
                            break # Only take the first one if override is enabled
                else:
                    for preset_category, presets_dict in normalized_presets.items():
                        selected_preset_name = kwargs.get(preset_category, 'None')
                        if selected_preset_name != "None" and selected_preset_name in presets_dict:
                            preset_text = presets_dict[selected_preset_name]
                            temp_components_for_llm.append(preset_text)
                
                # Include ALL individual adjustments (if not "None") for LLM context, regardless of weight
                for category_params in self.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED.values():
                    for param_name, weight_name, _ in category_params:
                        value = kwargs.get(param_name, 'None')
                        if value and value.strip().lower() != "none":
                            temp_components_for_llm.append(value.strip().lower()) # Just the value, not weighted format
                
                if temp_components_for_llm:
                    llm_additional_context = ", ".join(temp_components_for_llm)
                    
            llm_input_prompt = base_text_for_llm
            if llm_additional_context:
                llm_input_prompt += ", " + llm_additional_context

            if not llm_regenerate_on_each_run and self._last_llm_generated_prompt:
                current_master_prompt = self._last_llm_generated_prompt
            else:
                model_name = self.get_model_name(llm_model_selection)
                model_loaded = False
                if model_name:
                    model_loaded = self.load_model_optimized(model_name)

                if model_loaded and self.model is not None:
                    current_master_prompt = self.generate_with_model(
                        llm_input_prompt, llm_enhancement_style, llm_creativity_level, llm_max_length, prompt_type="positive"
                    )
                else:
                    current_master_prompt = self._create_fallback_enhancement(
                        llm_input_prompt, llm_enhancement_style, llm_creativity_level, prompt_type="positive"
                    )
                self._last_llm_generated_prompt = current_master_prompt

            if llm_purge_cache:
                self.cleanup_model()

        else: 
            current_master_prompt = base_text_for_llm
            self._last_llm_generated_prompt = "" 
        
        final_positive_prompt = current_master_prompt.strip()
        
        # Add non-LLM controlled components to the positive prompt
        components = []
        preset_components_raw = [] # Used for deduplication against individual adjustments

        normalized_presets, _, normalized_adjustable_details = self.get_normalized_options()

        if enable_preset_override_button:
            preset_category_order = list(normalized_presets.keys())
            for category_name in preset_category_order:
                selected_preset_name = kwargs.get(category_name, 'None')
                if selected_preset_name != "None" and selected_preset_name in normalized_presets[category_name]:
                    preset_text = normalized_presets[category_name][selected_preset_name]
                    components.append(preset_text)
                    preset_components_raw.append(preset_text.lower())
                    break
        else:
            for preset_category, presets_dict in normalized_presets.items():
                selected_preset_name = kwargs.get(preset_category, 'None')
                if selected_preset_name != "None" and selected_preset_name in presets_dict:
                    preset_text = presets_dict[selected_preset_name]
                    components.append(preset_text)
                    preset_components_raw.append(preset_text.lower())
            
            # Use the categorized param_definitions for iteration
            for category_params in self.ADJUSTABLE_DETAIL_OPTIONS_CATEGORIZED.values():
                for param_name, weight_name, _ in category_params:
                    value = kwargs.get(param_name, 'None')
                    weight = kwargs.get(weight_name, 1.0) 
                    formatted = self.weighted_format(value, weight) # Use weighted_format directly here
                    
                    should_add_param = True
                    if formatted:
                        formatted_lower = formatted.lower()
                        # Check if this formatted parameter is already covered by a selected preset
                        # This check is only relevant if LLM is NOT incorporating adjustments
                        if not (enable_llm_inference and incorporate_adjustments_into_llm):
                            for existing_preset_text in preset_components_raw:
                                if formatted_lower in existing_preset_text.lower(): 
                                    should_add_param = False
                                    break
                    
                    if should_add_param and formatted:
                        # Apply prefix/suffix here for non-LLM path
                        prefix_map = {
                            'animation_style': 'in ', 'aspect_ratio': 'in ', 'atmospheric_effects': 'featuring ',
                            'camera_angle': 'from ', 'camera_animation': 'with ', 'camera_lens': 'shot with ',
                            'camera_movement': 'with ', 'character_clothing': 'wearing ', 'character_expression': 'with a ',
                            'character_gender': 'a ', 'character_age': 'a ', 'character_ethnicity': 'a ',
                            'cinematic_technique': 'utilizing ', 'color_grading': 'with ', 'composition_rule': 'composed using ',
                            'depth_of_field': 'featuring ', 'environment_type': 'in ', 'film_emulation': 'emulating ',
                            'genre_influence': 'evoking ', 'image_quality': 'rendered at ', 'lens_effects': 'with ',
                            'lighting_color': 'in ', 'lighting_quality': 'with ', 'lighting_setup': 'using ',
                            'mood_emotion': 'with ', 'motion_blur': '', 'post_processing': 'processed with ',
                            'season': 'in ', 'shot_type': '', 'time_of_day': 'during ',
                            'visual_style': 'in ', 'weather_atmosphere': 'with ',
                        }
                        suffix_map = {
                            'animation_style': ' animation style', 'aspect_ratio': ' aspect ratio', 'atmospheric_effects': '',
                            'camera_angle': ' angle', 'camera_animation': ' camera animation', 'camera_lens': ' lens',
                            'camera_movement': ' camera movement', 'character_clothing': '', 'character_expression': ' expression',
                            'character_gender': ' character', 'character_age': ' character', 'character_ethnicity': ' character',
                            'cinematic_technique': ' technique', 'color_grading': ' color grading', 'composition_rule': '',
                            'depth_of_field': '', 'environment_type': ' setting', 'film_emulation': '',
                            'genre_influence': ' genre', 'image_quality': '', 'lens_effects': '',
                            'lighting_color': ' color temperature', 'lighting_quality': '', 'lighting_setup': '',
                            'mood_emotion': ' mood', 'motion_blur': '', 'post_processing': '',
                            'season': '', 'shot_type': ' shot', 'time_of_day': '',
                            'visual_style': ' style', 'weather_atmosphere': ' weather',
                        }
                        components.append(f"{prefix_map.get(param_name, '')}{formatted}{suffix_map.get(param_name, '')}")
            
        # This conditional ensures that manual components are appended ONLY IF
        # LLM inference is NOT enabled, OR if LLM is enabled but NOT incorporating adjustments.
        # If LLM is enabled AND incorporating adjustments, the LLM has already handled them.
        if components and (not enable_llm_inference or (enable_llm_inference and not incorporate_adjustments_into_llm)):
            components_text = ", ".join(components)
            if final_positive_prompt:
                final_positive_prompt += ", " + components_text
            else:
                final_positive_prompt = components_text
        final_positive_prompt = final_positive_prompt.replace(",,", ",").replace(" ,", ",").replace("  ", " ").strip()
        if final_positive_prompt and not final_positive_prompt.endswith((".", "!", "?")):
            final_positive_prompt += "."
        
        # --- Negative Prompt Generation ---
        generated_negative_prompt = self.generate_negative_prompt(negative_prompt_preset_name)
        
        # Store last generated negative prompt for reuse
        self._last_negative_prompt = generated_negative_prompt

        # Clean up model from VRAM if 'purge_cache' is enabled
        if llm_purge_cache:
            self.cleanup_model()

        # Return the actual results (UI update for text_output_display is removed)
        return (final_positive_prompt, generated_negative_prompt,)
