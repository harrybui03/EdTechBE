-- User and Course Seed Data
-- This script inserts 10 dummy users with various roles and types for testing and development.

DO $$
DECLARE
sys_manager_id UUID := uuid_generate_v4();
    admin_id UUID := uuid_generate_v4();
    moderator_id UUID := uuid_generate_v4();
    creator1_id UUID := uuid_generate_v4();
    creator2_id UUID := uuid_generate_v4();
    evaluator_id UUID := uuid_generate_v4();
    student1_id UUID := uuid_generate_v4();
    student2_id UUID := uuid_generate_v4();
    student3_id UUID := uuid_generate_v4();
    multi_role_user_id UUID := uuid_generate_v4();
BEGIN
    -- Insert 10 dummy users
INSERT INTO users (id, email, username, full_name, user_type, enabled) VALUES
                                                                           (sys_manager_id, 'chuthang2k4@gmail.com', 'sysmanager', 'System Manager', 'SYSTEM_USER', true),
                                                                           (admin_id, 'buithehuong03@gmail.com', 'adminuser', 'Admin User', 'SYSTEM_USER', true),
                                                                           (moderator_id, 'thangcdph36961@fpt.edu.vn', 'moduser', 'Moderator User', 'SYSTEM_USER', true),
                                                                           (creator1_id, 'mr.flooo1230@gmail.com', 'creator1', 'Course Creator One', 'WEBSITE_USER', true),
                                                                           (creator2_id, 'loduong4992@gmail.com', 'creator2', 'Course Creator Two', 'WEBSITE_USER', true),
                                                                           (evaluator_id, 'sora.gura.193@gmail.com', 'evaluator', 'Batch Evaluator', 'WEBSITE_USER', true),
                                                                           (student1_id, 'hoangbh2709@gmail.com', 'student1', 'Student One', 'WEBSITE_USER', true),
                                                                           (student2_id, 'duodaiz1412@gmail.com', 'student2', 'Student Two', 'WEBSITE_USER', true),
                                                                           (student3_id, 'duckie010203@gmail.com', 'student3', 'Student Three', 'WEBSITE_USER', true),
                                                                           (multi_role_user_id, 'huongbui@hachium.com', 'multirole', 'Multi Role User', 'WEBSITE_USER', true);

-- Assign roles to the dummy users
INSERT INTO user_roles (user_id, role) VALUES
                                           (sys_manager_id, 'SYSTEM_MANAGER'),
                                           (admin_id, 'SYSTEM_MANAGER'),       -- Another system manager for variety
                                           (moderator_id, 'MODERATOR'),
                                           (creator1_id, 'COURSE_CREATOR'),
                                           (creator2_id, 'COURSE_CREATOR'),
                                           (evaluator_id, 'BATCH_EVALUATOR'),
                                           (student1_id, 'LMS_STUDENT'),
                                           (student2_id, 'LMS_STUDENT'),
                                           (student3_id, 'LMS_STUDENT'),
                                           (multi_role_user_id, 'COURSE_CREATOR'), -- This user has two roles
                                           (multi_role_user_id, 'LMS_STUDENT');
END $$;

-- This migration seeds the database with a large, interconnected set of sample data
-- for courses, chapters, lessons, quizzes, and enrollments to facilitate development and testing.
-- This script has been updated to be compatible with the V1_0_0 consolidated schema.

DO $$
DECLARE
    -- User IDs
    instructor_id UUID;
    student1_id UUID;
    student2_id UUID;
    student3_id UUID;

    -- Course IDs (20 courses)
    course_ids UUID[] := array_agg(uuid_generate_v4()) FROM generate_series(1, 20);

    -- Chapter IDs (4 chapters per course for the first 5 courses)
    chapter_ids UUID[] := array_agg(uuid_generate_v4()) FROM generate_series(1, 20);

    -- Lesson IDs (2 lessons per chapter for the first 20 chapters)
    lesson_ids UUID[] := array_agg(uuid_generate_v4()) FROM generate_series(1, 40);

    -- Quiz IDs (1 quiz for the first 20 lessons)
    quiz_ids UUID[] := array_agg(uuid_generate_v4()) FROM generate_series(1, 20);

    -- Quiz Question IDs
    question_ids UUID[] := array_agg(uuid_generate_v4()) FROM generate_series(1, 40);

BEGIN
    -- 1. GET USER IDs FROM PREVIOUS MIGRATION
    SELECT id INTO instructor_id FROM users WHERE email = 'mr.flooo1230@gmail.com';
    SELECT id INTO student1_id FROM users WHERE email = 'hoangbh2709@gmail.com';
    SELECT id INTO student2_id FROM users WHERE email = 'duodaiz1412@gmail.com';
    SELECT id INTO student3_id FROM users WHERE email = 'duckie010203@gmail.com';

    -- 2. SEED COURSES (20 records)
    -- Note: 'published' column is removed, using 'status' instead.
INSERT INTO courses (
    id, title, slug, short_introduction, description,
    image, video_link, status, paid_course, course_price,
    selling_price, currency, amount_usd, enrollments,
    lessons, rating, language, target_audience,
    skill_level, learner_profile_desc
) VALUES
      (course_ids[1], 'Mastering Spring Boot 3', 'mastering-spring-boot-3', 'A comprehensive guide to building modern applications.', 'Covers dependency injection, microservices, security, and testing.', 'spring-boot-3.jpg', NULL, 'PUBLISHED', TRUE, 2500.00, 2200.00, 'VND', 2200.00, 1500, 45, 4.85, 'English', 'Backend Developers', 'Advanced', 'Professionals looking to master the latest Spring Boot features.'),
      (course_ids[2], 'Introduction to PostgreSQL', 'introduction-to-postgresql', 'Learn the fundamentals of the world''s most advanced open source database.', 'SQL basics, data types, and simple queries for beginners.', 'postgresql-intro.jpg', NULL, 'PUBLISHED', TRUE, 2000.00, 1800.00, 'VND', 1800.00, 850, 20, 4.60, 'English', 'Data Analysts and Beginners', 'Beginner', 'Anyone starting their journey with SQL and relational databases.'),
      (course_ids[3], 'Advanced Docker and Kubernetes', 'advanced-docker-kubernetes', 'Deploy and manage containerized applications at scale.', 'Deep dive into orchestration, networking, and storage.', 'docker-k8s.jpg', NULL, 'PUBLISHED', TRUE, 3000.00, 2800.00, 'VND', 2800.00, 1200, 60, 4.90, 'English', 'DevOps Engineers', 'Expert', 'Experienced users seeking to deploy and manage large-scale systems.'),
      (course_ids[4], 'React for Beginners', 'react-for-beginners', 'Build modern, interactive user interfaces with React.', 'Learn components, state, props, and hooks from scratch.', 'react-beginners.jpg', NULL, 'PUBLISHED', TRUE, 2300.00, 2000.00, 'VND', 2000.00, 2100, 35, 4.75, 'English', 'Front-end Developers', 'Beginner', 'Aspiring web developers and designers who want to learn React.'),
      (course_ids[5], 'Data Structures and Algorithms in Java', 'data-structures-algorithms-java', 'Master the essential concepts for coding interviews.', 'Covers arrays, linked lists, trees, graphs, and sorting algorithms.', 'dsa-java.jpg', NULL, 'PUBLISHED', TRUE, 2400.00, 2100.00, 'VND', 2100.00, 900, 50, 4.80, 'English', 'Software Engineers', 'Intermediate', 'Individuals preparing for technical interviews or strengthening their fundamentals.'),
      (course_ids[6], 'Building RESTful APIs with Node.js', 'restful-apis-nodejs', 'Create fast and scalable backend services.', 'Using Express.js, MongoDB, and best practices.', 'nodejs-api.jpg', NULL, 'DRAFT', TRUE, 2600.00, 2300.00, 'VND', 2300.00, 0, 0, 0.00, 'English', 'Backend Developers', 'Intermediate', 'Developers who want to build modern, production-ready APIs.'),
      (course_ids[7], 'Machine Learning with Python', 'machine-learning-python', 'An introduction to practical machine learning.', 'Using Scikit-learn, Pandas, and NumPy.', 'ml-python.jpg', NULL, 'DRAFT', TRUE, 2900.00, 2700.00, 'VND', 2700.00, 0, 0, 0.00, 'English', 'Data Scientists', 'Intermediate', 'Anyone looking to apply machine learning models to real-world data.'),
      (course_ids[8], 'CSS Grid and Flexbox', 'css-grid-flexbox', 'Modern CSS layout techniques explained.', 'Build complex, responsive layouts with ease.', 'css-layout.jpg', NULL, 'DRAFT', TRUE, 2000.00, 1800.00, 'VND', 1800.00, 0, 0, 0.00, 'English', 'Front-end Developers', 'Beginner', 'Web developers struggling with traditional layout methods.'),
      (course_ids[9], 'Go (Golang) Programming', 'go-programming-basics', 'Learn the basics of Google''s Go language.', 'Concurrency, channels, and building simple web servers.', 'golang-basics.jpg', NULL, 'DRAFT', TRUE, 2500.00, 2200.00, 'VND', 2200.00, 0, 0, 0.00, 'English', 'System Engineers', 'Beginner', 'Developers looking to learn a fast, concurrent, and modern language.'),
      (course_ids[10], 'Introduction to Cloud Computing', 'intro-to-cloud-computing', 'Understand the fundamentals of AWS, Azure, and GCP.', 'Virtual machines, storage, and serverless computing.', 'cloud-intro.jpg', NULL, 'DRAFT', TRUE, 2200.00, 2000.00, 'VND', 2000.00, 0, 0, 0.00, 'English', 'IT Professionals', 'Beginner', 'Anyone who needs a solid understanding of cloud service models and providers.'),
      (course_ids[11], 'Vue.js 3 Fundamentals', 'vuejs-3-fundamentals', 'Build reactive front-end applications with Vue.', 'Composition API, Vuex, and Vue Router.', 'vuejs-3.jpg', NULL, 'DRAFT', TRUE, 2500.00, 2200.00, 'VND', 2200.00, 0, 0, 0.00, 'English', 'Front-end Developers', 'Intermediate', 'Developers with basic JavaScript knowledge ready to learn a reactive framework.'),
      (course_ids[12], 'Advanced SQL Techniques', 'advanced-sql-techniques', 'Master window functions, CTEs, and query optimization.', 'For data analysts and backend developers.', 'advanced-sql.jpg', NULL, 'DRAFT', TRUE, 2800.00, 2600.00, 'VND', 2600.00, 0, 0, 0.00, 'English', 'Data Analysts and Backend Developers', 'Advanced', 'Users of SQL who want to write more efficient and complex queries.'),
      (course_ids[13], 'Test-Driven Development (TDD) in C#', 'tdd-in-csharp', 'Write robust and maintainable code with TDD.', 'Using NUnit and Moq.', 'tdd-csharp.jpg', NULL, 'DRAFT', TRUE, 2400.00, 2100.00, 'VND', 2100.00, 0, 0, 0.00, 'English', 'Software Engineers', 'Intermediate', 'C# developers aiming to improve code quality and testing practices.'),
      (course_ids[14], 'Introduction to UI/UX Design', 'intro-to-ui-ux-design', 'Principles of user-centric design.', 'Wireframing, prototyping, and user testing.', 'ui-ux-intro.jpg', NULL, 'DRAFT', TRUE, 2200.00, 2000.00, 'VND', 2000.00, 0, 0, 0.00, 'English', 'Designers and Developers', 'Beginner', 'Anyone interested in the process of creating user-friendly digital products.'),
      (course_ids[15], 'Cybersecurity Fundamentals', 'cybersecurity-fundamentals', 'Protecting systems from common threats.', 'Network security, cryptography, and ethical hacking.', 'cybersecurity.jpg', NULL, 'DRAFT', TRUE, 2900.00, 2700.00, 'VND', 2700.00, 0, 0, 0.00, 'English', 'IT Professionals', 'Intermediate', 'Individuals who want a foundational understanding of digital security.'),
      (course_ids[16], 'GraphQL for API Development', 'graphql-for-api-dev', 'Build flexible and efficient APIs with GraphQL.', 'Compared to REST, with Apollo Server.', 'graphql-api.jpg', NULL, 'DRAFT', TRUE, 2500.00, 2200.00, 'VND', 2200.00, 0, 0, 0.00, 'English', 'Backend Developers', 'Intermediate', 'Developers transitioning from REST to modern API technologies.'),
      (course_ids[17], 'Getting Started with Rust', 'getting-started-with-rust', 'Learn the basics of the Rust programming language.', 'Ownership, borrowing, and lifetimes.', 'rust-basics.jpg', NULL, 'DRAFT', TRUE, 2400.00, 2100.00, 'VND', 2100.00, 0, 0, 0.00, 'English', 'Systems Programmers', 'Beginner', 'Developers interested in high-performance, safe systems programming.'),
      (course_ids[18], 'DevOps CI/CD with Jenkins', 'devops-cicd-jenkins', 'Automate your build and deployment pipeline.', 'Jenkinsfiles, plugins, and best practices.', 'jenkins-cicd.jpg', NULL, 'DRAFT', TRUE, 2800.00, 2600.00, 'VND', 2600.00, 0, 0, 0.00, 'English', 'DevOps Engineers', 'Intermediate', 'Professionals looking to implement and manage automated delivery pipelines.'),
      (course_ids[19], 'Mobile App Development with Flutter', 'mobile-app-dev-flutter', 'Build cross-platform apps from a single codebase.', 'Dart, widgets, and state management.', 'flutter-mobile.jpg', NULL, 'DRAFT', TRUE, 2900.00, 2700.00, 'VND', 2700.00, 0, 0, 0.00, 'English', 'Mobile Developers', 'Intermediate', 'Developers aiming to build native-quality apps for iOS and Android quickly.'),
      (course_ids[20], 'The Complete Guide to Web Scraping', 'complete-guide-web-scraping', 'Extract data from websites using Python.', 'Beautiful Soup, Scrapy, and handling dynamic sites.', 'web-scraping.jpg', NULL, 'DRAFT', TRUE, 2500.00, 2200.00, 'VND', 2200.00, 0, 0, 0.00, 'English', 'Data Engineers', 'Intermediate', 'Anyone who needs to collect and process data from the public web.');
END $$;
