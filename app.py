import uuid
import re
import streamlit as st
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# -------------------- CONFIG --------------------
AZURE_OPENAI_ENDPOINT = "https://<your-endpoint>.openai.azure.com/"
AZURE_OPENAI_KEY = "<your-openai-key>"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-4o"

AZURE_SEARCH_ENDPOINT = "https://<your-search-endpoint>.search.windows.net"
AZURE_SEARCH_KEY = "<your-search-key>"
AZURE_SEARCH_INDEX = "<your-index-name>"
AZURE_SEARCH_VECTOR_FIELD = "text_vector"
AZURE_SEARCH_CONTENT_FIELD = "chunk"

# -------------------- CLIENTS --------------------
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-03-01-preview"
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# -------------------- VARIABLES --------------------
if "cart" not in st.session_state:
    st.session_state.cart = []

# -------------------- SYSTEM PROMPT --------------------
SYSTEM_PROMPT = (
    "You are Restobot, a restaurant assistant bot. "
    "Only answer using the provided context below — do not guess or make up information. "
    "Use function calling for actions like adding/removing/reviewing/placing orders. "
    "If unsure about a dish being on the menu, do not add it and ask the user to choose another item. "
)

# -------------------- SEARCH --------------------
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=[text],
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding

def search_restobot_index(query):
    embedding = get_embedding(query)
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=5,
        fields=AZURE_SEARCH_VECTOR_FIELD
    )
    results = search_client.search(
        search_text="",
        vector_queries=[vector_query]
    )
    return [doc[AZURE_SEARCH_CONTENT_FIELD] for doc in results]

# -------------------- ORDER FUNCTIONS --------------------
def parse_quantity_item(text):
    items = []
    parts = re.split(r'and|,', text)
    for part in parts:
        match = re.match(r'(\d+)?\s*(.*)', part.strip())
        if match:
            quantity = int(match.group(1)) if match.group(1) else 1
            item = match.group(2).strip().lower()
            items.append((item, quantity))
    return items

def menu_lookup(item_name, context_chunks):
    for chunk in context_chunks:
        if item_name.lower() in chunk.lower():
            price_match = re.search(rf"{item_name}.*?₹?(\d+)", chunk, re.IGNORECASE)
            price = int(price_match.group(1)) if price_match else 100
            return True, price
    return False, 0

def add_to_cart(item_text, context_chunks):
    items = parse_quantity_item(item_text)
    added_items = []
    not_found = []

    for item_name, qty in items:
        found, price = menu_lookup(item_name, context_chunks)
        if found:
            st.session_state.cart.append({"item": item_name, "quantity": qty, "price": price})
            added_items.append(f"{qty} x {item_name.title()} (₹{price} each)")
        else:
            not_found.append(item_name)

    response = ""
    if added_items:
        response += f"✅ Added to cart: {', '.join(added_items)}\n"
    if not_found:
        response += f"⚠️ Not found in menu: {', '.join(not_found)}\n"
    return response.strip()

def remove_from_cart(item_text):
    items = parse_quantity_item(item_text)
    removed_items = []

    for item_name, _ in items:
        for i in range(len(st.session_state.cart) - 1, -1, -1):
            if item_name in st.session_state.cart[i]["item"]:
                removed_items.append(st.session_state.cart[i]["item"])
                st.session_state.cart.pop(i)

    if removed_items:
        return f"🗑️ Removed from cart: {', '.join(removed_items)}"
    else:
        return "⚠️ None of the items were found in your cart."

def review_cart():
    if not st.session_state.cart:
        return "🛒 Your cart is empty."

    message = "🧾 **Your current order:**<br>"
    total = 0
    for item in st.session_state.cart:
        subtotal = item["quantity"] * item["price"]
        total += subtotal
        message += f"{item['quantity']} x {item['item'].title()} = ₹{subtotal}<br>"
    message += f"<br>**Total:** ₹{total}"

    return f"<div style='border: 1px solid #f0f2f6; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>{message}</div>"

def place_order():
    if not st.session_state.cart:
        return "🛍️ No items to place. Your cart is empty."

    order_summary = review_cart()
    st.session_state.cart.clear()
    return f"{order_summary}\n\n✅ Order placed! Your food will be prepared soon."

# -------------------- GPT-4O --------------------
def ask_gpt4o(query, context_chunks):
    if not context_chunks:
        return "Sorry, I couldn’t find that in the dataset."

    combined_context = "\n".join(context_chunks)
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{combined_context}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Restobot", page_icon="🍽️", layout="centered")

# Initialize session state
if "chat" not in st.session_state:
    st.session_state.chat = []

st.markdown("""
    <style>
    .stApp {
        background-color: #f2f7f5;
    }

    .chat-container {
        border: 2px solid #84c9ba;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        margin-top: 20px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .chat-message {
        border-radius: 15px;
        padding: 10px 15px;
        margin: 10px 0;
        max-width: 90%;
        font-size: 16px;
        line-height: 1.5;
        font-family: 'Segoe UI', sans-serif;
        word-wrap: break-word;
    }

    .user {
        background-color: #d1f2eb;
        align-self: flex-end;
        text-align: right;
        margin-left: auto;
        border: 1px solid #b2dfdb;
    }

    .assistant {
        background-color: #fce4ec;
        align-self: flex-start;
        text-align: left;
        margin-right: auto;
        border: 1px solid #f8bbd0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍽️ Your Restaurant Assistant")

# Chat input
user_input = st.chat_input("Type your message or order...")

# Handle user input and generate bot response
if user_input:
    # Step 1: Add user message
    st.session_state.chat.append({"role": "user", "content": user_input})
    user_input_lower = user_input.lower()

    # Step 2: Determine bot response
    if user_input_lower in ["hi", "hello", "hey", "hii"]:
        bot_reply = "Hello! I’m Restobot, your food assistant. 🍽️"
    elif "how are you" in user_input_lower:
        bot_reply = "I’m doing great and ready to take your order! 😊"
    elif "how can you help" in user_input_lower or "what can you do" in user_input_lower:
        bot_reply = "I can help you explore the menu and place food orders. 🍴"
    elif user_input_lower.startswith("add "):
        context = search_restobot_index("menu")
        bot_reply = add_to_cart(user_input[4:], context)
    elif user_input_lower.startswith("remove "):
        bot_reply = remove_from_cart(user_input[7:])
    elif any(phrase in user_input_lower for phrase in [
        "review order", "review my cart", "what's in my cart", 
        "order list", "display my order", "show my order", 
        "display my cart", "my order list", "what is in my cart", 
        "show order list", "cart items"
    ]):
        bot_reply = review_cart()
    elif "place order" in user_input_lower:
        bot_reply = place_order()
    else:
        context = search_restobot_index(user_input)
        bot_reply = ask_gpt4o(user_input, context)

    # Step 3: Add bot response
    st.session_state.chat.append({"role": "assistant", "content": bot_reply})

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for entry in st.session_state.chat:
    role = entry["role"]
    msg = entry["content"]
    if role == "user":
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; margin: 10px 0;">
                <div class='chat-message user'>{msg}</div>
                <div style="margin-left: 10px; font-size: 24px;">🧑‍💬</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; align-items: center; margin: 10px 0;">
                <div style="margin-right: 10px; font-size: 24px;">🤖</div>
                <div class='chat-message assistant'>{msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)
