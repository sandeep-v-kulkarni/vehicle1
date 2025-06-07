import streamlit as st
import boto3
import json
import base64
from PIL import Image
import io
import os
import uuid
import numpy as np
from typing import Dict, List, Tuple  # Add List here
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import threading
import queue
import re

class VehicleAnalyzer:
    def __init__(self):
        """Initialize the Bedrock client"""
        self.bedrock = boto3.client('bedrock-runtime')
        self.nova_canvas = "amazon.nova-canvas-v1:0"

    def generate_vehicle_image(self, vehicle_description: str) -> Image:
        """Generate vehicle image using Nova Canvas"""
        try:
            # Construct the prompt for Nova Canvas
            prompt = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": f"Create a realistic 3D rendering of a vehicle with these characteristics: {vehicle_description}. Show the vehicle from a 3/4 front view in a studio setting with professional lighting.",
                    "negativeText": "low quality, blurry, bad anatomy, distorted, deformed"
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "quality": "standard",
                    "height": 1024,
                    "width": 1024,
                    "cfgScale": 8.0
                }
            }
            
            # Invoke Nova Canvas
            response = self.bedrock.invoke_model(
                modelId=self.nova_canvas,
                body=json.dumps(prompt),
                contentType="application/json",
                accept="application/json"
            )
            
            # Parse the response
            response_body = json.loads(response.get('body').read())
            
            # Extract and decode the image
            if 'images' in response_body:
                image_data = base64.b64decode(response_body['images'][0])
                return Image.open(io.BytesIO(image_data))
            else:
                raise ValueError("No image data in response")
            
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None


    def analyze_aerodynamics(self, vehicle_desc: str) -> str:
        """Analyze aerodynamics using Claude"""
        try:
            prompt = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""As an automotive aerodynamics expert, analyze the following vehicle:

                        Vehicle Description: {vehicle_desc}

                        Provide a detailed analysis including:
                        1. Expected aerodynamic characteristics
                        2. Potential areas of concern
                        3. Estimated drag coefficient range
                        4. Recommendations for improvement
                        5. Impact on fuel efficiency/range

                        Format your response with clear sections and bullet points."""
                    }
                ],
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3
            }

            response = self.bedrock.invoke_model(
                modelId="us.anthropic.claude-3-opus-20240229-v1:0",
                body=json.dumps(prompt),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body['content'][0]['text']

        except Exception as e:
            st.error(f"Error in aerodynamic analysis: {str(e)}")
            return "Error generating aerodynamic analysis. Please try again."

    def analyze_ergonomics(self, vehicle_desc: str) -> str:
        """Analyze ergonomics using Claude"""
        try:
            prompt = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""As an automotive ergonomics expert, analyze the following vehicle:

                        Vehicle Description: {vehicle_desc}

                        Provide a detailed analysis including:
                        1. Expected ergonomic characteristics
                        2. Driver and passenger comfort considerations
                        3. Visibility and safety implications
                        4. Entry/exit ease assessment
                        5. Recommendations for improvement

                        Format your response with clear sections and bullet points."""
                    }
                ],
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3
            }

            response = self.bedrock.invoke_model(
                modelId="us.anthropic.claude-3-opus-20240229-v1:0",
                body=json.dumps(prompt),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body['content'][0]['text']

        except Exception as e:
            st.error(f"Error in ergonomic analysis: {str(e)}")
            return "Error generating ergonomic analysis. Please try again."

    def compare_designs(self, original_desc: str, modified_desc: str) -> str:
        """Compare two vehicle designs using Claude"""
        try:
            prompt = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Compare these two vehicle designs:

                        Original Design: {original_desc}
                        Modified Design: {modified_desc}

                        Provide a detailed comparison including:
                        1. Key differences in characteristics
                        2. Expected performance changes
                        3. Trade-offs between designs
                        4. Recommendations
                        5. Overall assessment

                        Format your response with clear sections and bullet points."""
                    }
                ],
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3
            }

            response = self.bedrock.invoke_model(
                modelId="us.anthropic.claude-3-opus-20240229-v1:0",
                body=json.dumps(prompt),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body['content'][0]['text']

        except Exception as e:
            st.error(f"Error in design comparison: {str(e)}")
            return "Error generating design comparison. Please try again."

class ReasoningAnalyzer:
    def __init__(self):
        """Initialize the Bedrock client for reasoning analysis"""
        self.client = boto3.client('bedrock-runtime')
        self.claude_model = "us.anthropic.claude-3-opus-20240229-v1:0"
        self.deepseek_model = "us.deepseek.r1-v1:0"
        self.model_id = self.deepseek_model  # Set default model_id to DeepSeek

    def analyze_modifications(self, original_prompt: str, aero_prompt: str, modified_image_base64: str = None) -> dict:
        """Analyze which aspects need evaluation based on the modifications"""
        try:
            # Create prompt with image reference if available
            image_context = ""
            if modified_image_base64:
                image_context = "\nI am also providing a generated 3D image of the modified vehicle for reference."
    
            prompt = f"""You are an automotive engineering expert. Analyze the following vehicle modifications and return ONLY a JSON response.
    
            Original Vehicle: {original_prompt}
            Modified Vehicle: {aero_prompt}{image_context}
    
            Task: Analyze which aspects need detailed evaluation.
    
            Instructions:
            1. Consider each aspect below
            2. Determine if it needs evaluation based on the modifications
            3. Provide a brief justification and relevant parameters
            4. Return ONLY a JSON object in exactly this format:
    
            {{
                "aspects": {{
                    "aerodynamics": {{
                        "needed": true,
                        "justification": "Brief explanation here",
                        "parameters": ["parameter1", "parameter2"]
                    }},
                    "ergonomics": {{
                        "needed": true,
                        "justification": "Brief explanation here",
                        "parameters": ["parameter1", "parameter2"]
                    }},
                    "safety": {{
                        "needed": true,
                        "justification": "Brief explanation here",
                        "parameters": ["parameter1", "parameter2"]
                    }},
                    "manufacturing": {{
                        "needed": true,
                        "justification": "Brief explanation here",
                        "parameters": ["parameter1", "parameter2"]
                    }},
                    "regulatory": {{
                        "needed": true,
                        "justification": "Brief explanation here",
                        "parameters": ["parameter1", "parameter2"]
                    }}
                }}
            }}
    
            Return ONLY the JSON object, no additional text or explanation."""
    
            # Correct request format for DeepSeek
            request_body = {
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["</s>"]
            }
    
            response = self.client.invoke_model(
                modelId=self.deepseek_model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
    
            response_body = json.loads(response.get('body').read())
            completion = response_body.get('generation', '')
    
            # Clean and parse the response
            try:
                # Find JSON content
                json_start = completion.find('{')
                json_end = completion.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = completion[json_start:json_end]
                    
                    # Clean the JSON string
                    json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                    json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
                    
                    # Parse JSON
                    result = json.loads(json_str)
                    
                    # Validate the structure
                    if 'aspects' not in result:
                        raise ValueError("Missing 'aspects' key in response")
                    
                    return result
                else:
                    return self._get_default_analysis()
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON: {str(e)}")
                return self._get_default_analysis()
                
        except Exception as e:
            st.error(f"Error in reasoning analysis: {str(e)}")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> dict:
        """Return default analysis structure if there's an error"""
        return {
            "aspects": {
                "aerodynamics": {
                    "needed": True,
                    "justification": "Default analysis needed",
                    "parameters": ["drag coefficient", "airflow patterns"]
                },
                "ergonomics": {
                    "needed": True,
                    "justification": "Default analysis needed",
                    "parameters": ["visibility", "comfort"]
                },
                "safety": {
                    "needed": True,
                    "justification": "Default analysis needed",
                    "parameters": ["crash performance", "structural integrity"]
                },
                "manufacturing": {
                    "needed": True,
                    "justification": "Default analysis needed",
                    "parameters": ["feasibility", "cost"]
                },
                "regulatory": {
                    "needed": True,
                    "justification": "Default analysis needed",
                    "parameters": ["compliance", "certification"]
                }
            }
        }

    def get_detailed_analysis(self, aspect: str, original_prompt: str, aero_prompt: str, 
                     parameters: List[str], modified_image_base64: str = None) -> str:
        """Get detailed analysis for selected aspect using Claude"""
        try:
            # Create prompt with image reference if available
            image_context = ""
            if modified_image_base64:
                image_context = "\nI am also providing a generated 3D image of the modified vehicle for reference."

            prompt = f"""Analyze the following aspect of vehicle modification:

            Original Vehicle: {original_prompt}
            Modified Vehicle: {aero_prompt}{image_context}

            Aspect for Analysis: {aspect}
            Parameters to Consider: {', '.join(parameters)}

            Provide a detailed analysis including:
            1. Technical Impact Assessment
            2. Quantitative Estimates (where possible)
            3. Potential Challenges and Risks
            4. Mitigation Strategies
            5. Industry Best Practices
            6. Recommendations

            Format the response with clear sections and bullet points."""

            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ] + ([{
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": modified_image_base64
                            }
                        }] if modified_image_base64 else [])
                    }
                ],
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3
            }

            response = self.client.invoke_model(
                modelId=self.claude_model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response.get('body').read())
            return response_body['content'][0]['text']

        except Exception as e:
            st.error(f"Error in detailed analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"


    
class VehicleImageGenerator:
    def __init__(self):
        """Initialize the Bedrock client"""
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = 'amazon.nova-canvas-v1:0'
        
        self.output_dir = 'generated_vehicles'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_image(self, prompt: str, negative_prompt: str = "low quality, blurry, bad anatomy") -> Tuple[Image.Image, str]:
        """Generate vehicle image using Nova Canvas"""
        try:
            body = json.dumps({
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": prompt,
                    "negativeText": negative_prompt
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "quality": "standard",
                    "height": 1024,
                    "width": 1024,
                    "cfgScale": 8.0
                }
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response.get("body").read())
            base64_image = response_body.get("images")[0]
            
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))
            
            filename = f"vehicle_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(self.output_dir, filename)
            image.save(filepath)
            
            return image, filepath
            
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None, None


def main():
    st.set_page_config(page_title="Comparative Vehicle Aerodynamic Analyzer", layout="wide")
    
    # Custom CSS for the title with lighter colors and white text
    st.markdown("""
        <style>
        .title-box {
            background: linear-gradient(270deg, #3498db, #74b9ff);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }
        
        .title-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: -50%;
            width: 150%;
            height: 100%;
            background: linear-gradient(
                90deg,
                rgba(255,255,255,0) 0%,
                rgba(255,255,255,0.2) 50%,
                rgba(255,255,255,0) 100%
            );
            transform: skewX(-20deg);
        }
        
        .title-text {
            color: white;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            margin: 0;
        }
        
        .subtitle-text {
            color: white;
            text-align: center;
            font-size: 1.2em;
            margin-top: 10px;
            font-family: 'Arial', sans-serif;
            opacity: 0.9;
        }

        .title-box:hover {
            background: linear-gradient(270deg, #2ecc71, #3498db);
            transition: background 0.3s ease;
        }
        </style>
        
        <div class="title-box">
            <h1 class="title-text">🚗 Vehicle Design Concept Exploration System</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize analyzers at the start
    analyzer = VehicleAnalyzer()
    reasoning_analyzer = ReasoningAnalyzer()
    
    # Initialize session state variables if they don't exist
    if 'family_image' not in st.session_state:
        st.session_state.family_image = None
    if 'aero_image' not in st.session_state:
        st.session_state.aero_image = None
    if 'family_analysis' not in st.session_state:
        st.session_state.family_analysis = None
    if 'aero_analysis' not in st.session_state:
        st.session_state.aero_analysis = None
    if 'aspects_analysis' not in st.session_state:
        st.session_state.aspects_analysis = None
    if 'vehicle_desc' not in st.session_state:
        st.session_state.vehicle_desc = ""
    if 'windshield_angle' not in st.session_state:
        st.session_state.windshield_angle = ""

    #st.title("🚗 Vehicle Design Analysis System")
    
    # Vehicle description input
    vehicle_desc = st.text_input(
        "Describe the vehicle characteristics:",
        value=st.session_state.vehicle_desc,
        placeholder="Example: A modern SUV with sleek design, intended for family use..."
    )
    # Initialize modified_image_base64 at the start
    modified_image_base64 = None

    if vehicle_desc:
        st.session_state.vehicle_desc = vehicle_desc
        
        # Generate initial image button
        if st.button("Generate Initial Vehicle") or st.session_state.family_image is not None:
            if st.session_state.family_image is None:
                with st.spinner("Generating vehicle visualization..."):
                    st.session_state.family_image = analyzer.generate_vehicle_image(vehicle_desc)
            
            if st.session_state.family_image:
                st.subheader("Initial Vehicle Design")
                st.image(st.session_state.family_image, caption="Generated Vehicle Design", use_container_width=True)
        
        # Windshield modification input
        windshield_angle = st.text_input(
            "Specify windshield angle modification:",
            value=st.session_state.windshield_angle,
            placeholder="Example: Increase windshield angle by 5 degrees..."
        )
        
        if windshield_angle:
            st.session_state.windshield_angle = windshield_angle
            
            # Generate modified vehicle button
            if st.button("Generate Modified Vehicle") or st.session_state.aero_image is not None:
                if st.session_state.aero_image is None:
                    with st.spinner("Generating modified vehicle..."):
                        st.session_state.aero_image = analyzer.generate_vehicle_image(
                            f"{vehicle_desc} with {windshield_angle}"
                        )
                
                if st.session_state.aero_image:
                    st.subheader("Modified Vehicle Design")
                    st.image(st.session_state.aero_image, caption="Modified Vehicle Design", use_container_width=True)
                    
                    # Convert the modified image to base64 and store in session state
                    if 'modified_image_base64' not in st.session_state:
                        modified_image_buffer = io.BytesIO()
                        st.session_state.aero_image.save(modified_image_buffer, format="PNG")
                        st.session_state.modified_image_base64 = base64.b64encode(modified_image_buffer.getvalue()).decode('utf-8')
                    
                    modified_image_base64 = st.session_state.modified_image_base64
                    
                    # Analyze modifications
                    if st.session_state.aspects_analysis is None:
                        with st.spinner("Analyzing modifications..."):
                            st.session_state.aspects_analysis = reasoning_analyzer.analyze_modifications(
                                vehicle_desc,
                                f"{vehicle_desc} with {windshield_angle}",
                                modified_image_base64
                            )
                    
                    if st.session_state.aspects_analysis:
                        st.subheader("🔍 Areas Requiring Evaluation")
                        
                        # Create a list of aspects that need evaluation
                        needed_aspects = []
                        for aspect, details in st.session_state.aspects_analysis['aspects'].items():
                            if details['needed']:
                                needed_aspects.append({
                                    'name': aspect,
                                    'details': details
                                })
                                
                                # Display aspect with justification
                                with st.expander(f"📊 {aspect.title()}", expanded=True):
                                    st.write("**Justification:**", details['justification'])
                                    st.write("**Parameters to analyze:**")
                                    for param in details['parameters']:
                                        st.write(f"- {param}")

                        if needed_aspects:
                            # Create columns for selection and analysis
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Let user select aspect for detailed analysis
                                selected_aspect = st.selectbox(
                                    "Select aspect for analysis:",
                                    options=[aspect['name'] for aspect in needed_aspects],
                                    format_func=lambda x: x.title()
                                )

                                if st.button("Generate Detailed Design Exploration"):
                                    # Store the new analysis in session state
                                    analysis_key = f"analysis_{selected_aspect}"
                                    if analysis_key not in st.session_state:
                                        with st.spinner(f"Generating detailed analysis for {selected_aspect}..."):
                                            selected_details = next(
                                                aspect['details'] for aspect in needed_aspects 
                                                if aspect['name'] == selected_aspect
                                            )
                                            
                                            st.session_state[analysis_key] = reasoning_analyzer.get_detailed_analysis(
                                                selected_aspect,
                                                vehicle_desc,
                                                f"{vehicle_desc} with {windshield_angle}",
                                                selected_details['parameters'],
                                                modified_image_base64
                                            )
                            
                            with col2:
                                # Display all stored analyses
                                for key in st.session_state.keys():
                                    if key.startswith("analysis_"):
                                        aspect_name = key.split("_")[1]
                                        with st.expander(f"Analysis: {aspect_name.title()}", expanded=True):
                                            st.markdown(st.session_state[key])

    # Add a reset button
    if st.button("Reset Design exploration"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()

