#!/usr/bin/env python3
"""
Test script for the Whisper STT server.
This script checks if the server is running and can accept requests.
"""

import requests
import sys

def test_health_check():
    """Test if the server is responding to health checks."""
    try:
        response = requests.get("http://localhost:8030/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed - Server is running")
            return True
        else:
            print(f"✗ Health check failed - Status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed - Error: {e}")
        return False

def test_api_endpoints():
    """Test if the API endpoints are available."""
    try:
        # Try to get the API docs
        response = requests.get("http://localhost:8030/docs", timeout=5)
        if response.status_code == 200:
            print("✓ API documentation is available at http://localhost:8030/docs")
            return True
        else:
            print(f"✗ API docs check failed - Status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ API docs check failed - Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Whisper STT Server on port 8030...\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        results.append(test_func())
    
    print("\n" + "="*50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("="*50)
    
    if all(results):
        print("\n✓ All tests passed! The Whisper server is ready to use.")
        print("\nYou can:")
        print("  - View API docs: http://localhost:8030/docs")
        print("  - View UI: http://localhost:8030/")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the server logs:")
        print("  docker compose logs -f")
        return 1

if __name__ == "__main__":
    sys.exit(main())
