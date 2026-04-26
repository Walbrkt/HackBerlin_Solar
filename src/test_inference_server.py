"""
Test Client for Reonic System Designer Inference Server

This script tests the FastAPI server with sample customer data.

Usage:
    1. Start the server in one terminal:
       python src/inference_server.py
    
    2. Run this test script in another terminal:
       python src/test_inference_server.py
"""

import requests
import json
import time
from typing import Dict, Any

# API endpoint
API_URL = "http://localhost:8000"
DESIGN_ENDPOINT = f"{API_URL}/api/design-system"
HEALTH_ENDPOINT = f"{API_URL}/health"


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}\n")


def test_health_check() -> bool:
    """Test the health check endpoint."""
    print_section("Testing Health Check")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            print(f"✓ Server is healthy")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {API_URL}")
        print(f"  Make sure the server is running: python src/inference_server.py")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_design_system(customer_data: Dict[str, Any]) -> bool:
    """Test the design-system endpoint with sample data."""
    print_section("Testing Design System Endpoint")
    
    print("Request Payload:")
    print(json.dumps(customer_data, indent=2))
    
    try:
        response = requests.post(
            DESIGN_ENDPOINT,
            json=customer_data,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Inference successful!")
            print(f"\nResponse:")
            print(json.dumps(result, indent=2))
            
            # Print formatted result
            print(f"\nSystem Design Recommendation:")
            print(f"  Solar Panels: {result['panels_needed']} panels")
            print(f"  Roof Space: {result['roof_space_sqm_needed']} m²")
            print(f"  Battery Storage: {result['recommended_battery_kwh']} kWh")
            print(f"  Estimated Cost: €{result['estimated_total_cost_euros']:,.0f}")
            
            return True
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {API_URL}")
        print(f"  Make sure the server is running: python src/inference_server.py")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def run_tests():
    """Run all tests with different customer profiles."""
    
    print_section("Reonic System Designer - Inference Server Test Suite")
    
    print("Testing connectivity...")
    time.sleep(1)
    
    # Test health check
    if not test_health_check():
        print("\n⚠️  Server is not running. Start it with:")
        print("   python src/inference_server.py")
        return
    
    # Test cases: Different customer profiles
    test_cases = [
        {
            "name": "Standard German Household (6 MWh/year, with EV)",
            "data": {
                "energy_demand_wh": 6000000,      # 6 MWh/year
                "has_ev": 1,                       # Electric vehicle
                "has_solar": 0,                    # No existing solar
                "has_storage": 0,                  # No existing battery
                "has_wallbox": 1,                  # Has wallbox for charging
                "house_size_sqm": 150,             # 150 m² house
                "heating_existing_electricity_demand_kwh": 500
            }
        },
        {
            "name": "Small Apartment (3 MWh/year, no EV, no heating)",
            "data": {
                "energy_demand_wh": 3000000,      # 3 MWh/year
                "has_ev": 0,                       # No EV
                "has_solar": 0,
                "has_storage": 0,
                "has_wallbox": 0,
                "house_size_sqm": 80,              # 80 m² apartment
                "heating_existing_electricity_demand_kwh": 0
            }
        },
        {
            "name": "Large House (10 MWh/year, with EV and heating)",
            "data": {
                "energy_demand_wh": 10000000,     # 10 MWh/year
                "has_ev": 1,
                "has_solar": 1,                    # Has existing solar (minor impact)
                "has_storage": 1,                  # Has existing battery
                "has_wallbox": 1,
                "house_size_sqm": 250,             # 250 m² house
                "heating_existing_electricity_demand_kwh": 2000
            }
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_section(f"Test Case {i}: {test_case['name']}")
        success = test_design_system(test_case['data'])
        results.append((test_case['name'], success))
        time.sleep(1)  # Brief pause between requests
    
    # Summary
    print_section("Test Summary")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    run_tests()
