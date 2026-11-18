# Handschrifterkennung (HTR) & Qualitäts-Benchmark

Zurück zur [README](../README.md)

## Grundprinzip & Parametrisierung

Da gehostete Vision-/Multimodal-Modelle (wie Google Gemini) nicht selbst nachtrainiert werden, wird die Erkennungsqualität über **System-Prompting und API-Parametrisierung** gesteuert (`temperature: 0.0`, `reasoning_effort: "low"` für Gemini, strukturierte JSON-Antwort in [`AiClient`](../src/main/java/de/in/fcb/ai/AiClient.java)):

* **1:1 Verbatim-Übernahme:** Rechtschreib-, Grammatik-, Groß-/Kleinschreibungs- und Zeichensetzungsfehler der Lernenden werden exakt übernommen und *nicht* stillschweigend korrigiert.
* **Streichungen ignorieren:** Sichtbar durchgestrichene Wörter oder Buchstaben werden nicht in den digitalen Text übernommen.
* **`[unleserlich]` statt Raten:** Bei echter Unleserlichkeit oder hoher Unsicherheit wird `[unleserlich]` an der exakten Textstelle markiert, statt ein Wort stillschweigend zu erraten oder zu glätten.

## Metriken (`CER`, `WER`, `Word Accuracy`)

Die Benchmark-Klasse [`HandwritingRecognitionBenchmarkTest`](../src/test/java/de/in/fcb/ai/HandwritingRecognitionBenchmarkTest.java) (via `-Pbenchmark-tests`, Metriken in [`HandwritingSimilarityMetrics`](../src/test/java/de/in/fcb/ai/HandwritingSimilarityMetrics.java)) misst die Abweichung zur manuell abgetippten Referenz-`.txt` über die Levenshtein-Editierdistanz:

* **CER (*Character Error Rate*):** Anteil fehlerhafter Zeichen (Einfügungen, Löschungen, Ersetzungen) an der Gesamtzeichenzahl der Referenz. Da ein geändertes Komma oder Leerzeichen nur 1 Zeichen kostet, ist die CER das stabilste Maß für die tatsächliche Lesegenauigkeit auf Buchstabenebene.
* **WER (*Word Error Rate*):** Anteil abweichender Wort-Tokens (whitespace-getrennt) an der Gesamtwortzahl der Referenz. Reagiert sehr empfindlich auf Interpunktions- und Leerzeichenvarianten – so zählt z. B. `(l.22)` vs. `(l. 22)` als 2 Wortfehler, obwohl jedes Zeichen korrekt erkannt wurde.
* **Word Accuracy:** 100 % − WER.

## Einordnung & Vergleich mit menschlichen Korrektoren

* **Menschliche Obergrenze (~95–97 %):** Bei freier Schülerhandschrift (inkl. Streichungen, Einschüben am Rand und individuellem Schriftbild) erreichen selbst zwei menschliche Korrektoren, die denselben 200-Wort-Schülertext unabhängig voneinander 1:1 abtippen, untereinander typischerweise nur **95–97 % Word Accuracy** (mehrdeutige `a`/`e`- oder `n`/`u`-Bögen, Zeilenumbrüche, kopfseitige Meta-Angaben).
* **Inhaltliche vs. sprachliche Bewertung:**
  * Für die **inhaltliche Bewertung** (Aufgabenverständnis, Argumentation, Textbelege) sind bereits **85–90 % Word Accuracy** unkritisch, da der Satz- und Textzusammenhang vollständig erhalten bleibt.
  * Für die **sprachliche Bewertung** (Rechtschreibung, Grammatik, Ausdruck) zählt neben den Quoten vor allem die Fehlerart: Leerzeichen in Zeilenverweisen oder Kopfbereich-Formatierungen sind unkritisch; entscheidend ist, dass echte Schülerfehler nicht durch stille Autokorrektur „geheilt“ werden.

## Qualitäts-Schwellwerte & Referenzwerte

Für den Benchmark hat sich eine Kombination aus drei Zielwerten bewährt:

| Kriterium | Zielwert | Bedeutung |
| --- | --- | --- |
| **Word Accuracy** | **≥ 90 %** (Ø ≥ 93 %) | Fängt ausgelassene Zeilen, Halluzinationen und grobe Aussetzer ab |
| **CER** | **≤ 3,5 %** (Ø ≤ 2,0 %) | Robustes Qualitätsmaß auf Buchstabenebene (unempfindlicher gegen Leerzeichen-Effekte) |
| **Fehler-Erhaltung** | **≥ 90 %** | Bekannte Schülerfehler im Text dürfen nicht stillschweigend autokorrigiert werden |

**Aktuelle Referenzwerte (`gemini-3.8-flash`, `reasoning_effort: "low"`, 8 EN/DE-Samples):**
**Word Accuracy 91,6 % – 100,0 %** (Ø **95,4 %**) · **CER 0,0 % – 3,8 %** (Ø **1,4 %**) · **WER 0,0 % – 8,4 %** (Ø **4,6 %**) · **~2,7–5,5 s pro Seite**.

## Benchmark ausführen

```bash
# Alle verfügbaren Samples im Standardverzeichnis auswerten:
FLAICHECKBOT_AI_API_KEY="..." mvn test -Pbenchmark-tests -Dfcb.ai.model=gemini-3.8-flash

# Optional einzelnes Subset & abweichendes Verzeichnis wählen:
FLAICHECKBOT_AI_API_KEY="..." mvn test -Pbenchmark-tests \
  -Dfcb.ai.model=gemini-3.8-flash \
  -Dfcb.samples.dir=/pfad/zu/exported_samples \
  -Dfcb.samples.filter=sample_1,sample_2,sample_9
```

Der vollständige Vergleichsbericht pro Sample wird anschließend nach `target/ocr-benchmark-report.md` geschrieben.
