import os
from PyPDF2 import PdfReader
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, pipeline

def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text, max_length=512):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def generate_questions(text, num_questions=5):
    model_name = "valhalla/t5-small-qa-qg-hl"
    model = TFT5ForConditionalGeneration.from_pretrained(model_name, from_pt=True)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    question_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    
    questions = []
    chunks = split_text(text, max_length=512)
    
    for chunk in chunks:
        qg_input = "generate questions: " + chunk
        result = question_generator(qg_input, max_length=128, num_return_sequences=1)
        questions.extend([res['generated_text'] for res in result])
    
    return questions[:num_questions]

def main():
    input_type = input("Enter 'text' for plain text or 'pdf' for PDF: ").strip().lower()
    
    if input_type == 'text':
        text = input("Enter your text: ").strip()
    elif input_type == 'pdf':
        pdf_path = input("Enter the path to your PDF file: ").strip()
        if not os.path.exists(pdf_path):
            print("The specified PDF file does not exist.")
            return
        text = extract_text_from_pdf(pdf_path)
    else:
        print("Invalid input. Please enter 'text' or 'pdf'.")
        return

    num_questions = int(input("Enter the number of questions to generate: ").strip())
    questions = generate_questions(text, num_questions)
    
    print("\nGenerated Questions:")
    for idx, question in enumerate(questions, 1):
        print(f"{idx}. {question}")

if __name__ == "__main__":
    main()
