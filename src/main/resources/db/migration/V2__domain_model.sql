CREATE TABLE school_class (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(80) NOT NULL UNIQUE,
    academic_year VARCHAR(20) NOT NULL,
    grade_level VARCHAR(20),
    foreign_language VARCHAR(30) NOT NULL DEFAULT 'EN',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE student (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(60) NOT NULL,
    display_alias VARCHAR(100),
    school_class_id UUID NOT NULL REFERENCES school_class(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_student_class_external UNIQUE (school_class_id, external_id)
);

CREATE TABLE test_definition (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(150) NOT NULL,
    foreign_language VARCHAR(30) NOT NULL DEFAULT 'EN',
    grade_level VARCHAR(20),
    learning_unit VARCHAR(100),
    source_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE test_task (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_definition_id UUID NOT NULL REFERENCES test_definition(id) ON DELETE CASCADE,
    position INT NOT NULL,
    title VARCHAR(120) NOT NULL,
    task_text TEXT NOT NULL,
    reference_text TEXT,
    grading_hints TEXT,
    max_points NUMERIC(6, 2) NOT NULL DEFAULT 10.00,
    CONSTRAINT uq_test_task_position UNIQUE (test_definition_id, position)
);

CREATE TABLE assignment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(150) NOT NULL,
    school_class_id UUID NOT NULL REFERENCES school_class(id) ON DELETE CASCADE,
    test_definition_id UUID NOT NULL REFERENCES test_definition(id) ON DELETE RESTRICT,
    exam_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE student_work (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assignment_id UUID NOT NULL REFERENCES assignment(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'NEW',
    recognized_text TEXT,
    grading_json TEXT,
    overall_feedback TEXT,
    achieved_points NUMERIC(6, 2) NOT NULL DEFAULT 0.00,
    max_points NUMERIC(6, 2) NOT NULL DEFAULT 0.00,
    prompt_tokens INT NOT NULL DEFAULT 0,
    completion_tokens INT NOT NULL DEFAULT 0,
    model_used VARCHAR(80),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_student_work_assignment_student UNIQUE (assignment_id, student_id)
);

CREATE TABLE student_work_page (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_work_id UUID NOT NULL REFERENCES student_work(id) ON DELETE CASCADE,
    page_number INT NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    mime_type VARCHAR(60) NOT NULL DEFAULT 'image/png',
    redaction_boxes_json TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_student_work_page_num UNIQUE (student_work_id, page_number)
);
