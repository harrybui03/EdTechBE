package com.example.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
public class BackendApplication {

	public static void main(String[] args) {
        ConfigurableApplicationContext context = SpringApplication.run(BackendApplication.class, args);

        String resolvedUser = context.getEnvironment().getProperty("spring.mail.username");
        String resolvedPass = context.getEnvironment().getProperty("spring.mail.password");
        System.out.println("Application.yml mail " + resolvedUser + "pass " + resolvedPass);

        String mailUser = System.getenv("SPRING_MAIL_USERNAME");
        String mailPass = System.getenv("SPRING_MAIL_PASSWORD");
        System.out.println("Environment mail " + mailUser + "pass " + mailPass);
	}

}
