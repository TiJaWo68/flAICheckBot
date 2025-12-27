-- Schema for flAICheckBot
-- Supports Classes, Students, and Training Data Management

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE, -- e.g. "2025KL8b"
    academic_year TEXT         -- e.g. "2025/2026"
);

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE,   -- e.g. "S12345" or specific school ID
    class_id INTEGER,
    FOREIGN KEY(class_id) REFERENCES classes(id)
);

-- Anonymous Training Data
CREATE TABLE IF NOT EXISTS training_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    original_text TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_set_id INTEGER,
    image_data BLOB NOT NULL,     -- The scanned handwriting stored directly
    mime_type TEXT,               -- e.g. "image/png"
    sample_text TEXT,             -- Line-level reference text (Specific segment)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(training_set_id) REFERENCES training_sets(id)
);

-- Evaluation Workflow (Test Templates)
CREATE TABLE IF NOT EXISTS test_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    grade_level TEXT,            -- e.g. "8"
    learning_unit TEXT,          -- e.g. "Lektion 1"
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS test_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER,
    position INTEGER,            -- Order of the task
    task_text TEXT,               -- The actual question/task
    reference_text TEXT,         -- Erwartungshorizont / Model Solution
    max_points INTEGER,
    resource_info TEXT,          -- e.g. "Text A" or "Audio B"
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(test_id) REFERENCES test_definitions(id)
);

-- Evaluation Workflow (Class/Student specific Execution)
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER,
    test_id INTEGER,              -- Reference to the template
    title TEXT NOT NULL,          -- Instance title, e.g. "Klausur vom 22.12."
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(class_id) REFERENCES classes(id),
    FOREIGN KEY(test_id) REFERENCES test_definitions(id)
);

CREATE TABLE IF NOT EXISTS student_work (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER,
    student_id INTEGER,
    image_data BLOB,
    recognized_text TEXT,
    grading_json TEXT,            -- Points achieved per task, teacher feedback
    status TEXT DEFAULT 'NEW',
    FOREIGN KEY(assignment_id) REFERENCES assignments(id),
    FOREIGN KEY(student_id) REFERENCES students(id)
);
