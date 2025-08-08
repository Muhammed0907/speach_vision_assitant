import requests
import argparse
import json
from urllib.parse import quote

API_BASE_URL = "https://manghe.shundaocehua.cn/screen/ai-assistant/detail/%7BmachineId%7D?machineId="
LISTEN_STATUS_URL = "https://manghe.shundaocehua.cn/screen/ai-assistant/aiSetting/"
# API_BASE_URL = "http://localhost:5000/products"

def fetch_product_by_name(machine_id):
    """Fetch product details by product name from the API"""
    try:
        # URL-encode the product name
        encoded_name = quote(machine_id)
        url = f"{API_BASE_URL}{encoded_name}"
        # print(url)
        
        response = requests.get(url, timeout=5)  # 5 second timeout
        # print(f"RES: {response.json()['code']}")
        if response.json()['code'] == 1:
            return {"error": "Product not found"}
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"error": "Product not found"}
        else:
            return {"error": f"API returned status code {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def check_listen_status(machine_id):
    """Check if the system should be listening by fetching isListen API"""
    try:
        # print()
        url = f"{LISTEN_STATUS_URL}{machine_id}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # print(f"data: {data}")
            # sys.exit(1)
            return {
                "code": data.get("code", 1),
                "data": data.get("data", False),
                "msg": data.get("msg", None),
                "isGreetGender": data.get("isGreetGender", False)
            }
        else:
            return {"code": 1, "data": False, "msg": f"HTTP {response.status_code}", "isGreetGender": False}
            
    except requests.exceptions.RequestException as e:
        return {"code": 1, "data": False, "msg": f"Connection error: {str(e)}", "isGreetGender": False}

def main():
    parser = argparse.ArgumentParser(description="Fetch product details by name")
    parser.add_argument("product_name", help="Name of the product to fetch")
    args = parser.parse_args()
    
    result = fetch_product_by_name(args.product_name)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Product Details:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()



    

