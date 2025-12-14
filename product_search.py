import json
import requests
from typing import Dict, List, Optional, Any
from openai import OpenAI


def search_similar_products(
    image_url: str,
    serpapi_api_key: str,
    openai_api_key: str
) -> Dict[str, List[Dict[str, Any]]]:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        
        attributes = extract_shopping_attributes(openai_client, image_url)
        if not attributes:
            return {"results": []}
        
        shopping_query = attributes.get("shopping_query", "")
        if not shopping_query:
            return {"results": []}
        
        serpapi_products = fetch_serpapi_products(serpapi_api_key, shopping_query)
        if not serpapi_products:
            return {"results": []}
        
        ranked_products = rank_products_visually(openai_client, image_url, serpapi_products)
        if not ranked_products:
            return {"results": []}
        
        sorted_products = sorted(ranked_products, key=lambda x: x["similarity_score"], reverse=True)
        top_3 = sorted_products[:3]
        
        return {"results": top_3}
        
    except Exception as e:
        return {"results": []}


def extract_shopping_attributes(openai_client: OpenAI, image_url: str) -> Optional[Dict[str, Any]]:
    prompt = """Analyze this product image and extract shopping attributes in JSON format:
{
    "category": "string",
    "product_keywords": ["string"],
    "colors": ["string"],
    "materials": ["string"],
    "style_keywords": ["string"],
    "shopping_query": "string (optimized Google Shopping search query)"
}

Return ONLY valid JSON, no other text."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        if not content:
            return None
        
        try:
            attributes = json.loads(content)
            return attributes
        except json.JSONDecodeError:
            return None
            
    except Exception:
        return None


def fetch_serpapi_products(serpapi_api_key: str, shopping_query: str) -> List[Dict[str, Any]]:
    endpoint = "https://serpapi.com/search"
    params = {
        "engine": "google_shopping",
        "q": shopping_query,
        "api_key": serpapi_api_key,
        "num": 20
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "error" in data:
            return []
        
        products = []
        
        shopping_results = data.get("shopping_results", [])
        for item in shopping_results:
            link = item.get("product_link") or item.get("link", "")
            thumbnail = item.get("serpapi_thumbnail") or item.get("thumbnail", "")
            
            product = {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "link": link,
                "thumbnail": thumbnail
            }
            
            if product["title"] and product["link"]:
                products.append(product)
        
        inline_results = data.get("inline_shopping_results", [])
        for item in inline_results:
            link = item.get("link", "")
            thumbnail = item.get("serpapi_thumbnail") or item.get("thumbnail", "")
            
            product = {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "link": link,
                "thumbnail": thumbnail
            }
            
            if product["title"] and product["link"]:
                products.append(product)
        
        categorized_results = data.get("categorized_shopping_results", [])
        for category in categorized_results:
            category_products = category.get("shopping_results", [])
            for item in category_products:
                link = item.get("product_link") or item.get("link", "")
                thumbnail = item.get("serpapi_thumbnail") or item.get("thumbnail", "")
                
                product = {
                    "title": item.get("title", ""),
                    "source": item.get("source", ""),
                    "link": link,
                    "thumbnail": thumbnail
                }
                
                if product["title"] and product["link"]:
                    products.append(product)
        
        return products[:20]
        
    except requests.RequestException:
        return []
    except Exception:
        return []


def rank_products_visually(
    openai_client: OpenAI,
    reference_image_url: str,
    products: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not products:
        return []
    
    valid_products = [p for p in products if p.get("thumbnail")]
    if not valid_products:
        return []
    
    prompt_parts = [
        "Compare the reference image (first image) with candidate product images below.",
        "Rank each product by visual similarity (0.0-1.0).",
        "",
        "Return ONLY JSON object with rankings array:",
        '{"rankings": [{"product_index": 0, "similarity_score": 0.85}, {"product_index": 1, "similarity_score": 0.72}]}',
        "",
        "Reference image:"
    ]
    
    image_contents = [
        {"type": "text", "text": "\n".join(prompt_parts)},
        {"type": "image_url", "image_url": {"url": reference_image_url}}
    ]
    
    for idx, product in enumerate(valid_products):
        image_contents.append({
            "type": "text",
            "text": f"\n--- Product {idx} ---\nTitle: {product['title']}\nSource: {product['source']}\nLink: {product['link']}"
        })
        if product["thumbnail"]:
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": product["thumbnail"]}
            })
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": image_contents
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.2
        )
        
        content = response.choices[0].message.content
        if not content:
            return []
        
        ranking_data = json.loads(content)
        
        rankings = ranking_data.get("rankings", [])
        if not rankings:
            rankings = ranking_data.get("products", [])
        
        ranked_map = {}
        for rank_item in rankings:
            idx = rank_item.get("product_index")
            if idx is not None and 0 <= idx < len(valid_products):
                ranked_map[idx] = float(rank_item.get("similarity_score", 0.0))
        
        results = []
        for idx, product in enumerate(valid_products):
            score = ranked_map.get(idx, 0.0)
            results.append({
                "brand": product.get("source", ""),
                "product_name": product.get("title", ""),
                "product_url": product.get("link", ""),
                "similarity_score": score
            })
        
        return results
        
    except json.JSONDecodeError:
        return []
    except Exception:
        return []
