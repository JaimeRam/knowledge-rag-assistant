-- Creates the langfuse_db database on first postgres startup.
-- The digimon user is a superuser (POSTGRES_USER), so it can create databases.
SELECT 'CREATE DATABASE langfuse_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse_db')\gexec
