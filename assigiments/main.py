
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
from dotenv import load_dotenv
import requests
import os
import rich # 🔹 For colored terminal output

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Provider setup
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 2: Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# Step 3: Run configuration
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# ✅ Tool to get products
@function_tool
def get_product_data() -> list:
    """
    Fetches product data from the API and returns each product's title, price, discounted price, discount percentage,
      description, image, is_new status, and tags
    """
    url = "https://template6-six.vercel.app/api/products/"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        products = []

        for item in data:
            price = item.get("price", 0)
            discount = item.get("dicountPercentage", 0)
            discounted_price = round(price * (1 - discount / 100), 2)

            product_info = {
                "title": item.get("title", "No title"),
                "price": price,
                "discounted_price": discounted_price,
                "discount_percentage": discount,
                "description": item.get("description", "No description"),
                "is_new": item.get("isNew", False),
                "image_url": item.get("Url", "No image"),
                "tags": item.get("tags", [])
            }
            products.append(product_info)

        return products

    except Exception as e:
        return {"error": str(e)}


# ✅ Agent Setup — Friendly style like screenshot
agent = Agent(
    name="shopping_agent",
    instructions=(
        "You are a friendly shopping assistant. When a user asks about products, use the get_product_data tool to fetch items. "
        "Your response should be in conversational language, as if you're chatting casually. "
        "Use phrases like 'I recommend', 'Here are a few suggestions', etc. "
        "Format product names in quotes. Use bullet points for listing multiple items. Avoid robotic tone — be elegant and warm."
    ),
    tools=[get_product_data],
)

# ✅ User Queries
product_related_queries = [
    "What are the products available in the store?",
    "Suggest something elegant for home decor.",
    "Can you show me rustic or vintage pieces for my living room?",
    "List all available products with price and discount details",
    "Show me the latest products in the store",
     "Are there any new arrivals?",
    "Can you show me products with specific tags?",
    "Do you have any cozy or comfy furniture recommendations?",
    "Which items are currently offering the biggest discount?",
    "What are the top-rated products?",
   "What products have the highest discount?",
   "What are the best-selling products?"
]

# ✅ Show responses in colored prompt/response format
for query in product_related_queries:
    rich.print(f"\n🧑‍💻 [bold cyan]User:[/] {query.strip()}")

    response = Runner.run_sync(
        agent,
        input=query,
        run_config=run_config
    )

    rich.print(f"\n🤖 [bold green]Agent:[/]\n{response.final_output.strip()}")
