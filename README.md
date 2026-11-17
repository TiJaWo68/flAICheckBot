# flAICheckBot

Web-Applikation zur KI-gestützten Korrektur und Auswertung von Fremdsprachen-Klassenarbeiten (Nachfolger des Java-Swing/Python-Prototyps [flAICheckBot-poc](https://github.com/TiJaWo68/flAICheckBot-poc)).

## Technik-Stack

* **Backend:** Java 25 (OpenJDK / Eclipse Temurin), Spring Boot 4
* **Frontend:** Vaadin Flow 25 (Full-Stack Java, Lumo Dark Theme)
* **Persistenz:** PostgreSQL, Spring Data JPA / Hibernate (`ddl-auto: none`, `open-in-view: false`)
* **Datenbank-Migrationen:** Flyway
* **Sicherheit:** Spring Security (lokales Form-Login mit BCrypt, optional Google OAuth2/OIDC)
* **KI-Anbindung:** OpenAI-kompatibler Client für Google AI Studio (Gemini), OpenAI und lokale Endpoints (Ollama)
* **Testing:** JUnit 5, Mockito, Testcontainers (PostgreSQL)
* **Logging:** Logback mit optionalem Logstash-JSON-Streaming über TCP
* **CI/CD:** GitHub Actions (CI Test-Suite & Release-Artifacts)

## Ersteinrichtung (Setup)

Solange noch kein Benutzer in `app_user` existiert, erzeugt die Anwendung beim Start einen einmaligen Setup-Token und gibt ihn im Server-Log aus:

```text
==============================================================
FLAICHECKBOT INITIAL SETUP TOKEN:  [ ABC23456 ]
Bitte diesen Token zur Freischaltung auf /setup eingeben.
==============================================================
```

1. Öffne `http://<host>:8080/setup`.
2. Vergib Benutzername, E-Mail, Passwort (min. 8 Zeichen) und den Setup-Token.
3. Nach erfolgreicher Einrichtung wird `/setup` dauerhaft gesperrt; unauthentifizierte Zugriffe landen auf `/login`.

## Anmeldung & Benutzerverwaltung

* Lokales Form-Login unter `/login` mit BCrypt-gehashten Passwörtern.
* Optionales Google OAuth2/OIDC erscheint automatisch, wenn `GOOGLE_CLIENT_ID` konfiguriert ist.
* Im Administrationsbereich (`/admin`) können nach dem Login lokale Benutzer angelegt, verwaltet und die KI-Verbindung (z. B. Google AI Studio API-Key & Modell) sowie der Systemstatus eingesehen werden.

## Konfiguration

| Umgebungsvariable | Default | Beschreibung |
| --- | --- | --- |
| `FLAICHECKBOT_DB_URL` | `jdbc:postgresql://localhost:5432/flaicheckbot` | JDBC-URL der PostgreSQL-Datenbank |
| `FLAICHECKBOT_DB_USER` | `flaicheckbot` | Datenbank-Benutzer |
| `FLAICHECKBOT_DB_PASSWORD` | `flaicheckbot` | Datenbank-Passwort |
| `FLAICHECKBOT_AI_API_KEY` | *leer* | Optionaler Fallback-API-Key für den KI-Endpoint |
| `GOOGLE_CLIENT_ID` | *leer* | Client-ID für Google-OAuth2-Login |
| `GOOGLE_CLIENT_SECRET` | *leer* | Client-Secret für Google-OAuth2-Login |
| `LOGGING_TCP_ENABLED` | `false` | Aktiviert strukturiertes JSON-Log-Streaming via TCP |
| `LOGGING_TCP_DESTINATIONS` | `${LOGGING_TCP_HOST}:${LOGGING_TCP_PORT}` | Komma-separierte Liste von Log-Zielen |
| `LOGGING_TCP_HOST` | `127.0.0.1` | Fallback-Host des TCP-Log-Receivers |
| `LOGGING_TCP_PORT` | `4560` | Fallback-Port des TCP-Log-Receivers |

## Build & Test

| Kommando | Beschreibung |
| --- | --- |
| `mvn clean test` | Führt alle Unit- und UI-Tests aus (`integration`-Tests übersprungen) |
| `mvn clean test -Pintegration-tests` | Führt zusätzlich alle Integrationstests mit Testcontainers (PostgreSQL) aus |
| `mvn clean package -Pproduction -DskipTests` | Baut das produktionsreife Fat-JAR inkl. Vaadin-Frontend-Bundle |
| `mvn spring-boot:run` | Startet die Anwendung lokal im Entwicklungsmodus |

## Remote-Log-Streaming (TCP)

Log-Events können in Echtzeit über einen TCP-Socket an einen externen Log-Receiver (z. B. Chainsaw, Lilith, Logstash) gestreamt werden. Die Ausgabe erfolgt strukturiert im Logstash-JSON-Format. Der Appender puffert asynchron und blockiert die Anwendung nicht, falls der Receiver nicht erreichbar ist; Verbindungsabbrüche werden automatisch durch Reconnects abgefangen.

Beispiel (einzelnes Ziel oder mehrere Ziele mit automatischem Failover):

```bash
# Mehrere Ziele (z. B. WLAN- und Docking-Station-IP):
LOGGING_TCP_ENABLED=true \
LOGGING_TCP_DESTINATIONS="192.168.178.62:4560,192.168.178.110:4560,192.168.178.135:4560" \
java -jar flaicheckbot.jar

# Oder klassisch mit einem einzelnen Ziel:
LOGGING_TCP_ENABLED=true \
LOGGING_TCP_HOST=192.168.178.62 \
LOGGING_TCP_PORT=4560 \
java -jar flaicheckbot.jar
```
