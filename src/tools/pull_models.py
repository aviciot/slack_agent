import os
import requests
import json
import time

# Get variables from the environment, with a fallback for local testing
ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
pull_models_env = os.environ.get("PULL_MODELS", "")

# Split the string of models into a list
models = [m.strip() for m in pull_models_env.split(',') if m.strip()]

if not models:
    print("No models specified in PULL_MODELS environment variable.")
    exit(1)

print("Starting model pull process...")
for model in models:
    try:
        # Check if the model already exists
        response = requests.post(f"{ollama_url}/api/show", data=json.dumps({"name": model}))
        
        if response.status_code == 200:
            print(f"Model '{model}' already exists. Skipping.")
            continue
        elif response.status_code == 404:
            # Model not found, proceed to pull
            pass
        else:
            # Handle other potential errors
            print(f"Unexpected status code for '{model}': {response.status_code}. Skipping.")
            continue

        # Pull the model
        data = {"name": model, "stream": True}
        print(f"Pulling model: {model}...")
        with requests.post(f"{ollama_url}/api/pull", data=json.dumps(data), stream=True) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        line_data = json.loads(line)
                        if "error" in line_data:
                            print(f"Error pulling {model}: {line_data['error']}")
                            break
                        if "status" in line_data:
                            print(line_data["status"])
                        else:
                            print(line_data)
                    except json.JSONDecodeError:
                        print(line.decode('utf-8'))
            print(f"Successfully pulled {model}")
    except requests.exceptions.RequestException as e:
        print(f"Error pulling model {model}: {e}")
    time.sleep(1) # Small delay to avoid hammering the server