import streamlit as st
from transformers import pipeline

# Load the pre-trained question answering model
qa_pipeline = pipeline("question-answering")

def main():
    st.title("Question Answering Model")

    # Input context
    context = st.text_area("Enter the context:", "", height=150)

    # Input question
    question = st.text_input("Enter your question:")

    if st.button("Predict"):
        if context.strip() and question.strip():
            # Perform question answering
            answers = qa_pipeline(question=question, context=context, topk=3)

            # Display the top 3 answers
            st.subheader("Top 3 Answers:")
            for i, ans in enumerate(answers):
                st.write(f"{i + 1}. {ans['answer']} (confidence: {ans['score']:.3f})")
        else:
            st.warning("Please enter both context and question.")

if __name__ == "__main__":
    main()
