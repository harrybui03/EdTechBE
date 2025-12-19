package com.example.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
public class BackendApplication {

	public static void main(String[] args) {
        String mailUser = System.getenv("SPRING_MAIL_USERNAME");
        String mailPass = System.getenv("SPRING_MAIL_PASSWORD");

        if (mailUser != null) System.setProperty("spring.mail.username", mailUser);
        if (mailPass != null) System.setProperty("spring.mail.password", mailPass);

		SpringApplication.run(BackendApplication.class, args);
	}

}
