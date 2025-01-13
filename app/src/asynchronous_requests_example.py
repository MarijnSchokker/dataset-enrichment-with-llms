import concurrent.futures

# Assume this function is in a separate library you can't modify or directly access.
def A(input_string):
    # Simulate a long-running API call
    from time import sleep
    from random import randint
    seconds = randint(1, 10)
    sleep(seconds)  # Simulate delay
    return f"Processed in {seconds}s: {input_string}"

# Function to handle subprocesses
def B(all_inputs):
    results = []

    # Use ProcessPoolExecutor to run in separate subprocesses
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        future_to_input = {executor.submit(A, input_str): input_str for input_str in all_inputs}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_input):
            input_str = future_to_input[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {input_str}: {e}")

    return results

# Example usage
if __name__ == "__main__":
    input_data = ["hello", "world", "foo", "bar", "baz", "qux"]
    output = B(input_data)
    print(output)