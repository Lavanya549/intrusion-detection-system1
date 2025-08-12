-- Drop the table if it exists

-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS cyber;

-- Switch to the created database
USE cyber;

-- Create the user table
CREATE TABLE `users` (
    `name` VARCHAR(225),
    `email` VARCHAR(225),
    `password` VARCHAR(225),
    `Address` VARCHAR(225)
    )