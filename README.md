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
| `mvn clean test` | Führt alle Unit- und UI-Tests aus (`e2e`- und `benchmark`-Tests übersprungen) |
| `mvn clean test -Pintegration-tests` | Führt zusätzlich alle Integrationstests (ohne `e2e` und `benchmark`) aus |
| `mvn test -Pproduction,e2e-tests` | Führt die Playwright-Browser-E2E-Workflow-Tests aus |
| `mvn test -Pbenchmark-tests` | Führt den Handschrift-OCR-Benchmark (`HandwritingRecognitionBenchmarkTest`) gegen Referenz-Samples aus und schreibt `target/ocr-benchmark-report.md` |
| `mvn clean package -Pproduction -DskipTests` | Baut das produktionsreife Fat-JAR inkl. Vaadin-Frontend-Bundle |
| `mvn spring-boot:run` | Startet die Anwendung lokal im Entwicklungsmodus |

## Handschrifterkennung (HTR) & Qualitäts-Benchmark

### Grundprinzip & Parametrisierung

Da gehostete Vision-/Multimodal-Modelle (wie Google Gemini) nicht selbst nachtrainiert werden, wird die Erkennungsqualität über **System-Prompting und API-Parametrisierung** gesteuert (`temperature: 0.0`, `reasoning_effort: "low"` für Gemini, strukturierte JSON-Antwort):

* **1:1 Verbatim-Übernahme:** Rechtschreib-, Grammatik-, Groß-/Kleinschreibungs- und Zeichensetzungsfehler der Lernenden werden exakt übernommen und *nicht* stillschweigend korrigiert.
* **Streichungen ignorieren:** Sichtbar durchgestrichene Wörter oder Buchstaben werden nicht in den digitalen Text übernommen.
* **`[unleserlich]` statt Raten:** Bei echter Unleserlichkeit oder hoher Unsicherheit wird `[unleserlich]` an der exakten Textstelle markiert, statt ein Wort stillschweigend zu erraten oder zu glätten.

### Metriken (`CER`, `WER`, `Word Accuracy`)

Die Benchmark-Klasse `HandwritingRecognitionBenchmarkTest` (via `-Pbenchmark-tests`, Metriken in `HandwritingSimilarityMetrics`) misst die Abweichung zur manuell abgetippten Referenz-`.txt` über die Levenshtein-Editierdistanz:

* **CER (*Character Error Rate*):** Anteil fehlerhafter Zeichen (Einfügungen, Löschungen, Ersetzungen) an der Gesamtzeichenzahl der Referenz. Da ein geändertes Komma oder Leerzeichen nur 1 Zeichen kostet, ist die CER das stabilste Maß für die tatsächliche Lesegenauigkeit auf Buchstabenebene.
* **WER (*Word Error Rate*):** Anteil abweichender Wort-Tokens (whitespace-getrennt) an der Gesamtwortzahl der Referenz. Reagiert sehr empfindlich auf Interpunktions- und Leerzeichenvarianten – so zählt z. B. `(l.22)` vs. `(l. 22)` als 2 Wortfehler, obwohl jedes Zeichen korrekt erkannt wurde.
* **Word Accuracy:** 100 % − WER.

### Einordnung & Vergleich mit menschlichen Korrektoren

* **Menschliche Obergrenze (~95–97 %):** Bei freier Schülerhandschrift (inkl. Streichungen, Einschüben am Rand und individuellem Schriftbild) erreichen selbst zwei menschliche Korrektoren, die denselben 200-Wort-Schülertext unabhängig voneinander 1:1 abtippen, untereinander typischerweise nur **95–97 % Word Accuracy** (mehrdeutige `a`/`e`- oder `n`/`u`-Bögen, Zeilenumbrüche, kopfseitige Meta-Angaben).
* **Inhaltliche vs. sprachliche Bewertung:**
  * Für die **inhaltliche Bewertung** (Aufgabenverständnis, Argumentation, Textbelege) sind bereits **85–90 % Word Accuracy** unkritisch, da der Satz- und Textzusammenhang vollständig erhalten bleibt.
  * Für die **sprachliche Bewertung** (Rechtschreibung, Grammatik, Ausdruck) zählt neben den Quoten vor allem die Fehlerart: Leerzeichen in Zeilenverweisen oder Kopfbereich-Formatierungen sind unkritisch; entscheidend ist, dass echte Schülerfehler nicht durch stille Autokorrektur „geheilt“ werden.

### Qualitäts-Schwellwerte & Referenzwerte

Für den Benchmark hat sich eine Kombination aus drei Zielwerten bewährt:

| Kriterium | Zielwert | Bedeutung |
| --- | --- | --- |
| **Word Accuracy** | **≥ 90 %** (Ø ≥ 93 %) | Fängt ausgelassene Zeilen, Halluzinationen und grobe Aussetzer ab |
| **CER** | **≤ 3,5 %** (Ø ≤ 2,0 %) | Robustes Qualitätsmaß auf Buchstabenebene (unempfindlicher gegen Leerzeichen-Effekte) |
| **Fehler-Erhaltung** | **≥ 90 %** | Bekannte Schülerfehler im Text dürfen nicht stillschweigend autokorrigiert werden |

**Aktuelle Referenzwerte (`gemini-3.8-flash`, `reasoning_effort: "low"`, 8 EN/DE-Samples):**
**Word Accuracy 91,6 % – 100,0 %** (Ø **95,4 %**) · **CER 0,0 % – 3,8 %** (Ø **1,4 %**) · **WER 0,0 % – 8,4 %** (Ø **4,6 %**) · **~2,7–5,5 s pro Seite**.

```bash
# Beispielaufruf des Benchmarks (API-Key über Umgebungsvariable):
FLAICHECKBOT_AI_API_KEY="..." mvn test -Pbenchmark-tests -Dfcb.ai.model=gemini-3.8-flash
```

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
