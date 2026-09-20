-- Person table: one row per individual
CREATE TABLE IF NOT EXISTS person (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recognition_count INTEGER DEFAULT 1
);

-- Embeddings table: multiple embeddings per person (different angles)
CREATE TABLE IF NOT EXISTS face_embedding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    label TEXT DEFAULT 'frontal',  -- e.g., frontal, left, right, up, down
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_face_embedding_person_id ON face_embedding(person_id);