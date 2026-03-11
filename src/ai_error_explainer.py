error = result.stderr

# fast rule-based explanation
explanation = quick_explain(error)

# if unknown error → use AI
if explanation is None:

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": f"Explain this Python error briefly:\n{error}"
            }
        ],
        options={"num_predict":120}
    )

    explanation = response["message"]["content"]