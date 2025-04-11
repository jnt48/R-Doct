from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import statistics
import json
import os
from datetime import datetime
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Document Management and Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configure Google Generative AI with your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Gemini functionality will not work.")

# Configure the Gemini model
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini model: {e}")
    model = None

# Data Models
class CollectionsData(BaseModel):
    collections: Dict[str, List[Dict[str, Any]]]

class AnalysisResult(BaseModel):
    insights: List[str]
    qualityMetrics: Dict[str, float]
    recommendations: List[str]
    collectionSpecificInsights: Dict[str, List[str]] = {}
    dataDistribution: Dict[str, Any] = {}
    fieldFrequencyAnalysis: Dict[str, Any] = {}

# Image Processing Functions
def input_image_setup(file_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(file_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()

        image_data = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_bytes).decode()
        }
        return image_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

def get_gemini_response(prompt: str, image_content: dict) -> str:
    if not model:
        raise HTTPException(status_code=500, detail="Gemini model not configured")
    
    response = model.generate_content([prompt, image_content])
    return response.text

# Document Collection Analysis Functions
def analyze_field_frequency(collections):
    """Analyze field frequency across collections"""
    field_analysis = {}
    
    for collection_name, docs in collections.items():
        field_counts = {}
        total_docs = len(docs)
        
        if total_docs == 0:
            continue
            
        # Count occurrences of each field
        for doc in docs:
            if "data" in doc and isinstance(doc["data"], dict):
                for field in doc["data"].keys():
                    if field in field_counts:
                        field_counts[field] += 1
                    else:
                        field_counts[field] = 1
        
        # Calculate percentage for each field
        field_stats = {}
        for field, count in field_counts.items():
            field_stats[field] = {
                "count": count,
                "percentage": round((count / total_docs) * 100, 1)
            }
            
        field_analysis[collection_name] = field_stats
    
    return field_analysis

def analyze_data_distribution(collections):
    """Analyze data distribution across collections"""
    distribution = {
        "totalDocuments": 0,
        "collectionBreakdown": {},
        "dateDistribution": {}
    }
    
    total_docs = 0
    date_counts = {}
    
    for collection_name, docs in collections.items():
        doc_count = len(docs)
        total_docs += doc_count
        
        # Add to collection breakdown
        distribution["collectionBreakdown"][collection_name] = {
            "count": doc_count,
            "percentage": 0  # Will calculate after total is known
        }
        
        # Analyze creation dates if available
        for doc in docs:
            if "createdAt" in doc and doc["createdAt"] and "seconds" in doc["createdAt"]:
                date_str = datetime.fromtimestamp(doc["createdAt"]["seconds"]).strftime("%Y-%m")
                if date_str in date_counts:
                    date_counts[date_str] += 1
                else:
                    date_counts[date_str] = 1
    
    # Update total documents
    distribution["totalDocuments"] = total_docs
    
    # Calculate percentages for collection breakdown
    if total_docs > 0:
        for collection_name in distribution["collectionBreakdown"]:
            count = distribution["collectionBreakdown"][collection_name]["count"]
            distribution["collectionBreakdown"][collection_name]["percentage"] = round((count / total_docs) * 100, 1)
    
    # Sort date distribution by date
    sorted_dates = sorted(date_counts.items())
    distribution["dateDistribution"] = {date: count for date, count in sorted_dates}
    
    return distribution

def generate_collection_specific_insights(collections):
    """Generate specific insights for each collection"""
    collection_insights = {}
    
    for collection_name, docs in collections.items():
        insights = []
        doc_count = len(docs)
        
        if doc_count == 0:
            insights.append(f"No documents found in the {collection_name} collection.")
            collection_insights[collection_name] = insights
            continue
        
        # Add basic count insight
        insights.append(f"Contains {doc_count} documents.")
        
        # Analyze field completeness
        field_completeness = {}
        common_fields = set()
        
        # First pass: identify all fields
        for doc in docs:
            if "data" in doc and isinstance(doc["data"], dict):
                for field in doc["data"].keys():
                    common_fields.add(field)
        
        # Second pass: check completeness
        for field in common_fields:
            field_count = 0
            non_empty_count = 0
            
            for doc in docs:
                if "data" in doc and isinstance(doc["data"], dict):
                    if field in doc["data"]:
                        field_count += 1
                        if doc["data"][field]:  # Check if non-empty
                            non_empty_count += 1
            
            if field_count > 0:
                completeness = (non_empty_count / field_count) * 100
                field_completeness[field] = completeness
        
        # Add insights about field completeness
        if field_completeness:
            least_complete = min(field_completeness.items(), key=lambda x: x[1])
            most_complete = max(field_completeness.items(), key=lambda x: x[1])
            
            if least_complete[1] < 70:
                insights.append(f"The '{least_complete[0]}' field is only {round(least_complete[1])}% complete.")
            
            insights.append(f"The '{most_complete[0]}' field is {round(most_complete[1])}% complete.")
        
        # Check for timestamps
        has_timestamps = False
        for doc in docs[:5]:  # Check first 5 docs
            if "createdAt" in doc and doc["createdAt"]:
                has_timestamps = True
                break
                
        if not has_timestamps:
            insights.append("Documents lack creation timestamps, which limits temporal analysis.")
        
        collection_insights[collection_name] = insights
    
    return collection_insights

async def analyze_with_gemini(collections, collection_counts, total_documents, field_analysis):
    """Use Gemini model to analyze collection data and generate insights"""
    
    # Prepare a summary of the collections for Gemini
    collection_summary = []
    for name, docs in collections.items():
        # Create a summary of each collection with sample data
        # Limit the number of documents and fields to avoid token limits
        doc_samples = []
        for i, doc in enumerate(docs[:3]):  # Sample up to 3 documents
            if "data" in doc and isinstance(doc["data"], dict):
                # Format the document data
                doc_data = {k: v for k, v in doc["data"].items() if k != "rawText"}
                doc_samples.append(doc_data)
        
        collection_summary.append({
            "name": name,
            "document_count": len(docs),
            "sample_documents": doc_samples,
            "field_frequency": field_analysis.get(name, {})
        })
    
    # Create prompt for Gemini
    prompt = f"""
    Analyze the following document collections from a database:
    
    {json.dumps(collection_summary, indent=2)}
    
    Total documents: {total_documents}
    Collection distribution: {collection_counts}
    
    Provide the following analysis in JSON format:
    1. "insights": A list of 7 specific insights about the data, patterns, quality, and potential improvements.
       Make these insights specific, actionable, and based on the actual data provided.
    2. "qualityMetrics": A dictionary with these metrics (values should be integers between 0-100):
       - dataCompleteness: How complete the data fields are
       - formatConsistency: How consistent the data format is
       - metadataQuality: Quality of metadata and classification
       - extractionAccuracy: How accurate the data extraction appears to be
       - dataStructure: How well-structured the data is
       - searchability: How easily the data can be searched and retrieved
       - overallQuality: Overall quality score
    3. "recommendations": List of 6 actionable recommendations to improve the document management system
       focusing on structure, completeness, accuracy, and usability.
    
    Return ONLY valid JSON output without any additional text or explanation.
    """
    
    try:
        # Generate response from Gemini
        response = model.generate_content(prompt)
        
        # Parse the JSON response
        try:
            # The response might include markdown code blocks or other formatting
            # Try to extract just the JSON part
            content = response.text
            
            # If the response is wrapped in code blocks, extract just the JSON
            if "```json" in content and "```" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            analysis_data = json.loads(content)
            
            # Ensure we have all required fields
            if not all(k in analysis_data for k in ["insights", "qualityMetrics", "recommendations"]):
                raise ValueError("Missing required fields in Gemini response")
                
            # Convert any float metrics to integers
            analysis_data["qualityMetrics"] = {
                k: round(v) if isinstance(v, (int, float)) else 0 
                for k, v in analysis_data["qualityMetrics"].items()
            }
            
            return AnalysisResult(**analysis_data)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Response text: {response.text}")
            # Fallback to basic analysis
            raise ValueError("Could not parse Gemini response")
            
    except Exception as e:
        print(f"Error using Gemini for analysis: {e}")
        # Fallback to basic analysis
        insights = generate_basic_insights(collections, collection_counts, total_documents)
        quality_metrics = calculate_basic_quality_metrics(collections)
        recommendations = generate_basic_recommendations(collections, quality_metrics)
        
        return AnalysisResult(
            insights=insights,
            qualityMetrics=quality_metrics,
            recommendations=recommendations
        )

def generate_basic_insights(collections, collection_counts, total_documents):
    """Generate basic insights without using AI"""
    insights = []
    
    # Collection distribution insights
    if collection_counts:
        most_common = max(collection_counts.items(), key=lambda x: x[1])
        insights.append(
            f"The most common document type is '{most_common[0]}' with {most_common[1]} documents "
            f"({round(most_common[1]/total_documents*100, 1)}% of total)."
        )
    
    # Check for field consistency across collections
    all_fields = {}
    for name, docs in collections.items():
        for doc in docs:
            if "data" in doc and isinstance(doc["data"], dict):
                for field in doc["data"].keys():
                    if field in all_fields:
                        all_fields[field] += 1
                    else:
                        all_fields[field] = 1
    
    # Find most common fields
    if all_fields:
        common_fields = sorted(all_fields.items(), key=lambda x: x[1], reverse=True)[:3]
        common_fields_str = ", ".join([f"'{field}'" for field, _ in common_fields])
        insights.append(f"The most common fields across all documents are {common_fields_str}.")
    
    # Add collection-specific insights
    for name, count in collection_counts.items():
        percentage = (count / total_documents) * 100
        insights.append(f"'{name}' documents make up {round(percentage, 1)}% of your database.")
    
    # Add general insights
    insights.append(f"Your database contains {total_documents} documents across {len(collections)} collections.")
    
    if total_documents > 10:
        insights.append("Consider implementing more structured metadata to improve searchability and analysis capabilities.")
    
    return insights[:7]  # Limit to top 7 insights

def calculate_basic_quality_metrics(collections):
    """Calculate basic quality metrics without using AI"""
    metrics = {
        "dataCompleteness": 0,
        "formatConsistency": 0, 
        "metadataQuality": 0,
        "extractionAccuracy": 0,
        "dataStructure": 0,
        "searchability": 0,
        "overallQuality": 0
    }
    
    if not collections:
        return metrics
    
    # Simple algorithm to estimate data completeness
    completeness_scores = []
    consistency_scores = []
    field_types = {}
    
    for collection_name, docs in collections.items():
        collection_fields = {}
        
        for doc in docs:
            if "data" in doc and isinstance(doc["data"], dict):
                fields = doc["data"]
                if fields:
                    # Count non-empty fields
                    non_empty = sum(1 for v in fields.values() if v)
                    completeness = (non_empty / max(1, len(fields))) * 100
                    completeness_scores.append(completeness)
                    
                    # Track fields for consistency analysis
                    for field, value in fields.items():
                        if field not in collection_fields:
                            collection_fields[field] = []
                        
                        # Track field type
                        value_type = type(value).__name__
                        if field not in field_types:
                            field_types[field] = {}
                        if value_type in field_types[field]:
                            field_types[field][value_type] += 1
                        else:
                            field_types[field][value_type] = 1
                        
                        # Add non-empty values to collection
                        if value:
                            collection_fields[field].append(value)
        
        # Calculate consistency for each field
        for field, values in collection_fields.items():
            if len(values) >= 2:
                # Simple consistency score: if all types are the same
                field_type_counts = field_types.get(field, {})
                total_values = sum(field_type_counts.values())
                if total_values > 0:
                    max_type_count = max(field_type_counts.values())
                    type_consistency = (max_type_count / total_values) * 100
                    consistency_scores.append(type_consistency)
    
    # Set metrics based on available data
    if completeness_scores:
        metrics["dataCompleteness"] = round(sum(completeness_scores) / len(completeness_scores))
    else:
        metrics["dataCompleteness"] = 50  # Default value
    
    if consistency_scores:
        metrics["formatConsistency"] = round(sum(consistency_scores) / len(consistency_scores))
    else:
        metrics["formatConsistency"] = 60  # Default value
    
    # Calculate metadata quality
    has_timestamps = False
    has_ids = False
    
    for collection_name, docs in collections.items():
        for doc in docs:
            if "createdAt" in doc and doc["createdAt"]:
                has_timestamps = True
            if "id" in doc:
                has_ids = True
            if has_timestamps and has_ids:
                break
    
    metadata_score = 50
    if has_timestamps:
        metadata_score += 25
    if has_ids:
        metadata_score += 25
    
    metrics["metadataQuality"] = metadata_score
    
    # Set other metrics based on what we've calculated
    metrics["extractionAccuracy"] = max(0, min(100, metrics["formatConsistency"] - 5))
    metrics["dataStructure"] = max(0, min(100, (metrics["dataCompleteness"] + metrics["formatConsistency"]) / 2))
    metrics["searchability"] = max(0, min(100, (metrics["dataCompleteness"] + metrics["metadataQuality"]) / 2))
    
    # Calculate overall quality
    metrics["overallQuality"] = round((
        metrics["dataCompleteness"] * 0.25 +
        metrics["formatConsistency"] * 0.2 +
        metrics["metadataQuality"] * 0.2 +
        metrics["extractionAccuracy"] * 0.15 +
        metrics["dataStructure"] * 0.1 +
        metrics["searchability"] * 0.1
    ))
    
    return metrics

def generate_basic_recommendations(collections, quality_metrics):
    """Generate basic recommendations without using AI"""
    recommendations = []
    
    # Recommend based on quality metrics
    if quality_metrics["dataCompleteness"] < 70:
        recommendations.append(
            "Improve data completeness by implementing validation rules for essential fields."
        )
    
    if quality_metrics["metadataQuality"] < 70:
        recommendations.append(
            "Enhance metadata quality by adding standardized taxonomies and classification schemes."
        )
    
    if quality_metrics["formatConsistency"] < 70:
        recommendations.append(
            "Standardize data formats across documents to improve consistency and searchability."
        )
    
    # Add general recommendations
    recommendations.append(
        "Implement a regular data quality assessment process to track improvements over time."
    )
    
    recommendations.append(
        "Consider using advanced machine learning techniques to automate document classification."
    )
    
    recommendations.append(
        "Develop a feedback loop with users to identify areas where extraction accuracy can be improved."
    )
    
    return recommendations

# API Endpoints for Collection Analysis
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_collections(data: CollectionsData):
    try:
        collections = data.collections
        
        # Calculate basic statistics
        collection_counts = {name: len(docs) for name, docs in collections.items()}
        total_documents = sum(collection_counts.values())
        
        if total_documents == 0:
            raise HTTPException(status_code=400, detail="No documents found in collections")
        
        # Add field frequency analysis
        field_analysis = analyze_field_frequency(collections)
        
        # Add data distribution analysis
        data_distribution = analyze_data_distribution(collections)
        
        # Use Gemini for advanced analysis if available
        if model and GOOGLE_API_KEY:
            results = await analyze_with_gemini(collections, collection_counts, total_documents, field_analysis)
            
            # Add our additional analyses
            results.fieldFrequencyAnalysis = field_analysis
            results.dataDistribution = data_distribution
            results.collectionSpecificInsights = generate_collection_specific_insights(collections)
            
            return results
        else:
            # Fallback to basic analysis if Gemini is not available
            insights = generate_basic_insights(collections, collection_counts, total_documents)
            quality_metrics = calculate_basic_quality_metrics(collections)
            recommendations = generate_basic_recommendations(collections, quality_metrics)
            collection_specific_insights = generate_collection_specific_insights(collections)
            
            return AnalysisResult(
                insights=insights,
                qualityMetrics=quality_metrics,
                recommendations=recommendations,
                collectionSpecificInsights=collection_specific_insights,
                fieldFrequencyAnalysis=field_analysis,
                dataDistribution=data_distribution
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# API Endpoints for Document Extraction
@app.post("/extract")
async def extract_data(
    file: UploadFile = File(...),
    instructions: str = Form("")
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        file_bytes = await file.read()
        image_content = input_image_setup(file_bytes)

        extraction_prompt = (
            "You are an expert document data extractor. Analyze the provided image of a document of any type "
            "(e.g., form, invoice, letter, etc.) and identify all relevant fields and information. "
            "Return only the extracted data in a valid JSON format with descriptive keys and no additional text or markdown formatting. "
            "Do not include any triple backticks or code fences. Include no \\n in the code"
        )
        if instructions.strip():
            extraction_prompt += "\nAdditional instructions: " + instructions.strip()

        extracted_data = get_gemini_response(extraction_prompt, image_content)
        return {"extracted_data": extracted_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy API Endpoints (maintained for backward compatibility)
@app.get("/collections")
async def get_collections():
    """Return a list of collections from the database - maintained for backward compatibility"""
    return {"collections": ["users", "documents", "metadata"]}

@app.get("/analysis")
async def get_analysis():
    """Return pre-calculated analysis of database collections"""
    # Sample data for backward compatibility
    return {
        "users": {
            "document_count": 120,
            "field_analysis": {
                "name": {"count": 120, "percentage": 100},
                "email": {"count": 118, "percentage": 98.3},
                "role": {"count": 98, "percentage": 81.7}
            },
            "sample_document": {"name": "John Doe", "email": "john@example.com", "role": "user"}
        },
        "documents": {
            "document_count": 342,
            "field_analysis": {
                "title": {"count": 342, "percentage": 100},
                "content": {"count": 338, "percentage": 98.8},
                "author": {"count": 290, "percentage": 84.8}
            },
            "sample_document": {"title": "Sample Document", "content": "Content here...", "author": "Jane Smith"}
        }
    }

@app.get("/collections/{collection_id}")
async def get_collection(collection_id: str):
    """Return data for a specific collection - maintained for backward compatibility"""
    # Sample data for backward compatibility
    return {
        "name": collection_id,
        "documents": [
            {
                "id": "doc1",
                "createdAt": {"seconds": 1680123456},
                "data": {"title": "Sample", "content": "This is sample content"}
            },
            {
                "id": "doc2",
                "createdAt": {"seconds": 1680123789},
                "data": {"title": "Another Sample", "content": "More sample content"}
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)