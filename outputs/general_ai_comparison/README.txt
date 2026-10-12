GENERAL-PURPOSE AI COMPARISON — Instructions
=============================================

Purpose:
  Test whether general-purpose AI systems (ChatGPT, Claude) can reliably
  classify images as AI-generated vs real, compared with our purpose-built
  FusionDetector.

Folder contents:
  comparison_images/      — 10 test images
  comparison_template.csv — Results template (our model columns filled in)
  README.txt              — This file

Standardized prompt (use EXACTLY this for every image, for both systems):
----------------------------------------------------------------------

  Is this image AI-generated or a real-world/camera-captured image?

  Give:
  1. Your classification: AI-generated or Real
  2. Your confidence from 0-100%

  Do not use external image-search or provenance information.
  Judge only the image itself.

----------------------------------------------------------------------

Procedure:
  1. Upload each image from comparison_images/ one at a time.
  2. Paste the standardized prompt above.
  3. Record the system's classification and confidence in the CSV.
  4. Do NOT provide any hints about the ground truth or our model's answer.
  5. Use the same prompt for every image. Do not modify it.
  6. If the system refuses to answer or hedges, record what it says in "notes".

Image distribution:
  3 REAL SD1.5    (in-distribution for our model)
  2 FAKE SD1.5    (in-distribution for our model)
  1 REAL ADM      (held-out generator for our model)
  2 FAKE ADM      (held-out generator for our model)
  1 REAL Wukong   (held-out generator for our model)
  1 FAKE Wukong   (held-out generator for our model)

Ground truth labels are known and recorded in the CSV.
Images were selected from the Experiment 4 test split, not cherry-picked
based on our model's predictions.
