import psycopg2
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException
from pydantic import BaseModel

app = FastAPI()

# 🔗 Connect to Laravel’s Postgres
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="gLdv4DObzlTii1RV",
    host="db.zavcdgxjqbkpsafishmg.supabase.co",
    port="5432"
)
cursor = conn.cursor()

# 🔹 Stub: replace with CLIP later
def get_embedding(image_bytes: bytes) -> np.ndarray:
    return np.random.rand(512)

# 🟢 Add product + store embedding
@app.post("/add_product/")
async def add_product(
    product_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = None
):
    try:
        image_bytes = await image.read()
        embedding = get_embedding(image_bytes)

        cursor.execute("""
            INSERT INTO product_embeddings (product_id, name, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (product_id) DO UPDATE
            SET name = EXCLUDED.name,
                embedding = EXCLUDED.embedding
        """, (product_id, name, psycopg2.Binary(pickle.dumps(embedding))))

        conn.commit()
        return {"message": "Product added successfully", "product_id": product_id}

    except Exception as e:
        conn.rollback()  # 👈 reset transaction state
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# 🟢 Request body for recommendations
class RecommendRequest(BaseModel):
    product_id: str
    top_k: int = 5

# 🟢 Recommend similar products
@app.post("/recommend/")
async def recommend_items(req: RecommendRequest):
    # Get target embedding
    cursor.execute("SELECT embedding FROM product_embeddings WHERE product_id = %s", (str(req.product_id),))
    row = cursor.fetchone()
    if not row:
        return {"error": "Product not found"}

    target_embedding = pickle.loads(row[0])

    # Get all other products
    cursor.execute("SELECT product_id, name, embedding FROM product_embeddings WHERE product_id != %s", (str(req.product_id),))
    rows = cursor.fetchall()

    similarities = []
    for pid, name, emb_blob in rows:
        emb = pickle.loads(emb_blob)
        sim = np.dot(target_embedding, emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(emb))
        similarities.append((pid, name, float(sim)))

    # Sort & return top_k
    similarities.sort(key=lambda x: x[2], reverse=True)
    recommendations = [
        {"product_id": pid, "name": name, "similarity": sim}
        for pid, name, sim in similarities[:req.top_k]
    ]
    
    print("Looking for product_id:", req.product_id)

    return {"recommendations": recommendations}
