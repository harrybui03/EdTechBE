package com.example.backend.dto.response.rag;

import lombok.Data;

import java.util.List;

@Data
public class AskRAGResponse {
    private String answer;
    private List<Source> sources; // Can be CourseSource or LessonSource

    @Data
    public static class Source {
        // Common fields
        private String title;
        private String slug;
        
        // Lesson-specific fields (null for course sources)
        private String lessonId;
        private String lessonTitle;
        private String courseId;
        private String courseTitle;
        private String courseSlug;
        private String docType; // "transcript" (primary) or "lesson" (fallback) or "course_overview"
    }
    
    // Deprecated: Use Source instead
    @Deprecated
    @Data
    public static class CourseSource {
        private String title;
        private String slug;
    }
}

