# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrates the multi-stage critique workflow for VTO results."""

from models.gemini import analyze_images_with_gemini


def run_vto_critique_workflow(apparel_image_gcs: str, result_image_gcs: str) -> dict:
    """Runs the full VTO critique workflow, starting with placement analysis.

    Args:
        apparel_image_gcs: The GCS URI of the source garment image.
        result_image_gcs: The GCS URI of the final generated VTO image.

    Returns:
        A dictionary containing the results of the analysis stages.
    """
    critique_data = {}

    # --- STAGE 1: Placement Analysis --- 
    try:
        placement_prompt = (
            "You are a fashion critic. Analyze the two provided images. "
            "The first is a source garment, the second is that garment on a person. "
            "Does the apparel image appear to be well placed on the model? "
            "Provide a brief, one-sentence critique."
        )
        placement_critique = analyze_images_with_gemini(
            prompt=placement_prompt,
            image_uris=[apparel_image_gcs, result_image_gcs]
        )
        critique_data["placement_analysis"] = placement_critique
    except Exception as e:
        print(f"Error during VTO placement analysis: {e}")
        critique_data["placement_analysis"] = "Could not analyze image placement."

    # --- STAGE 2: Pose Analysis (Future Implementation) --- 
    # pose_result = analyze_pose(result_image_gcs)
    # critique_data["pose_analysis"] = pose_result

    # --- STAGE 3: Style Critique with Gemini + Pose Data (Future Implementation) --- 
    # style_critique = analyze_style_with_gemini(..., pose_data=pose_result)
    # critique_data["style_critique"] = style_critique

    return critique_data
