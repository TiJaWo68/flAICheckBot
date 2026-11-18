# Betrieb, Remote-Log-Streaming & Datenmigration

Zurück zur [README](../README.md)

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

## Bestandsdaten-Migration (SQLite PoC)

Im Administrationsbereich unter `/admin` steht der Reiter **Datenmigration** ([`DataMigrationTab`](../src/main/java/de/in/fcb/ui/admin/DataMigrationTab.java) / [`PocMigrationService`](../src/main/java/de/in/fcb/migration/PocMigrationService.java)) zur Verfügung, um SQLite-Datenbanken (`flaicheckbot.db`, max. 200 MB) aus dem Java-Swing/Python-Prototyp zweistufig zu übernehmen:

1. **Hochladen & Vorschau:** Zeigt vor dem Schreibvorgang Klassen, Schüler, Klassenarbeiten und vorhandene Abgaben tabellarisch an.
2. **Selektiver Import:** Klassen, Tests, Zuordnungen inkl. Scans sowie optional vorhandene OCR-Texte werden idempotent nach PostgreSQL übertragen, sodass bereits importierte Datensätze bei wiederholtem Lauf übersprungen werden.
