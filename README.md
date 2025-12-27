# flAICheckBot - Foreign Language AI Check Bot

**flAICheckBot** soll Fremdsprachenlehrer bei der Korrektur von Arbeiten unterstützen. Das System kombiniert eine Handschriften-KI (ICR) mit LLM-basierten Analysen.

## Status: Funktionaler Prototyp / Proof-of-Concept
Das Projekt befindet sich in einer fortgeschrittenen Prototyp-Phase. Die Kern-Architektur steht und wesentliche Workflows sind funktional, jedoch fehlen noch viele Feature-Implementierungen und umfassende Praxistests.

## Kernfunktionen
- **Intelligente Handschriften-KI (ICR):** Lokale TrOCR-basierte Engine, die durch integriertes Fine-Tuning (LoRA) direkt in der App auf individuelle Schülermerkmale trainiert werden kann.
- **Leistungsstarker Batch-Editor:** Import von Scans mit integriertem Editor für Zuschnitt, Schwärzung und Bildoptimierung.
- **Vielseitiger Dokumenten-Import:** Unterstützung für Aufgabenstellungen in den Formaten PDF, DOCX, DOC, ODT, RTF, TXT und Markdown.
- **Enterprise-ready Architektur:**
    - **Java 21 + FlatLaf:** Moderne, reaktionsschnelle UI mit Dark Mode.
    - **SQLite:** Lokale, verschlüsselte (optional) Speicherung von Bilddaten und Bewertungen.
    - **Log4j2:** Professionelles Logging mit täglicher Archivierung und rollierenden ZIP-Files.
    - **Parallele Initialisierung:** Schneller Programmstart durch asynchrone Backend-Prozesse.

## Workflows

### 1. KI-Training (Bereich Training)
Optimiert die Erkennungsrate für spezifische Handschriften oder Dokumententypen.
- **Import:** Bilder von Schülerarbeiten werden importiert und ggf. im Batch-Editor zugeschnitten.
- **Annotierung:** Der Lehrer gibt den korrekten Referenztext (Ground Truth) für die Bildausschnitte ein.
- **Training:** Start des LoRA Fine-Tunings. Das Modell lernt die individuellen Merkmale der Handschrift, um zukünftige Erkennungen zu verbessern.

### 2. Test-Definition (Bereich Test-Definition)
Erstellt die Grundlage für eine automatisierte Korrektur.
- **Metadaten:** Festlegung von Titel, Klassenstufe und Lerneinheit.
- **Aufgaben-Import:** Einlesen der Aufgabenstellung aus PDF- oder Word-Dokumenten.
- **Strukturierung:** Aufteilung des Tests in einzelne Aufgaben inkl. Vergabe von Maximalpunktzahlen und Hinterlegung von Musterlösungen/Referenztexten.

### 3. Bewertung & Korrektur (Bereich Evaluation)
Der operative Kern für die tägliche Korrekturarbeit.
- **Laden:** Auswahl eines Tests und Import der Schülerarbeiten. Die Lade-Animation gibt Feedback über den Fortschritt.
- **Erkennung & Grading:** Die KI erkennt den geschriebenen Text und vergleicht ihn mit der Test-Definition, um einen Score- und Feedback-Vorschlag zu generieren.
- **Review:** Manuelle Korrektur der KI-Vorschläge. Über die globale Schriftgrößen-Synchronisation und Maximierungs-Features lässt sich die Ansicht optimal anpassen.
- **Status-Management:** Markieren von korrigierten Arbeiten als "Bewertet". Über Filter-Optionen behält man den Überblick über noch ausstehende Aufgaben.
- **Export:** Abschluss der Korrektur durch Generierung eines Ergebnisberichts (z.B. PDF).

## Systemvoraussetzungen
- **Java:** Version 21 oder höher.
- **Python:** Version 3.12+ (für die KI-Engine).
- **GPU:** Empfohlen (NVIDIA mit CUDA-Support), CPU-Zweitmodus automatisch möglich.

## Installation & Setup

### 1. KI-Engine (Python-Backend)
Die KI-Engine läuft als lokaler FastAPI-Service.
```bash
cd src/ai
# Virtual Environment erstellen
python3 -m venv venv
source venv/bin/activate  # venv\Scripts\activate unter Windows
# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 2. Java-Anwendung (Maven)
```bash
# Kompilieren und Abhängigkeiten laden
mvn clean package
# Anwendung starten
java -jar target/flAICheckBot.jar
```

## Installation unter Windows 11 (Automatisch)

Der flAICheckBot ist darauf ausgelegt, so einfach wie möglich gestartet zu werden. Unter Windows 11 erkennt das Programm beim ersten Start automatisch, ob eine passende Python-Umgebung vorhanden ist.

1.  **Dateien vorbereiten:** Entpacken Sie das Programm in ein Verzeichnis Ihrer Wahl (empfohlen: `%LocalAppData%\flAICheckBot`).
2.  **Starten:** Starten Sie die Anwendung (z.B. über `flAICheckBot.exe` oder `mvn exec:java`).
3.  **Automatisches Setup:** Wenn kein Python gefunden wird, lädt der Bot automatisch eine portable Version von Python sowie alle notwendigen KI-Bibliotheken (ca. 1GB) im Hintergrund herunter. Der Fortschritt wird in der Statusleiste angezeigt.

> [!TIP]
> Sie müssen nichts manuell konfigurieren. Stellen Sie lediglich sicher, dass Sie beim ersten Start eine stabile Internetverbindung haben.

### Manuelle Installation (Optional)
Falls Sie eine eigene Python-Umgebung nutzen möchten:
1. Erstellen Sie ein `.venv` im Projekt-Root.
2. Installieren Sie die Abhängigkeiten: `pip install -r src/ai/requirements.txt`.
3. Setzen Sie ggf. die Umgebungsvariable `FL_KI_PYTHON` auf den Pfad Ihres Python-Interpreters.

## Projektstruktur
- `src/main/java`: Java-Anwendungslogik (Swing UI, DB-Management, AI-Bridge).
- `src/ai`: Python-Backend mit TrOCR-Modell, Training-Loop und API-Server.
- `log/`: Automatisch generierte System-Logs.

---

