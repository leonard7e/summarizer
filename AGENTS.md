## Grundlegende Richtlinien

### Sprache
* Der generierte Code, Kommentare sowie die Dokumentation müssen auf Englisch verfasst sein.

### Dokumentation
Es wird strikt unterschieden zwischen:
* **User-documentation**: Einstiegspunkt ist `doc/user_manual.adoc` (Installation, Quickstart, CLI-Bedienung, Konfiguration).
* **Developer-documentation**: Einstiegspunkt ist `doc/development_manual.adoc` (Architektur, interne Details).

**Aufbau**: Die Einstiegsdateien sind minimal zu halten und enthalten ausschließlich AsciiDoc-Attribute und `include::`-Direktiven. Inhalte gehören in thematisch getrennte Dateien unter `doc/user/` bzw. `doc/dev/`. Neue Inhalte niemals direkt in die Einstiegsdatei schreiben — fehlt eine passende Datei, eine neue anlegen und über `include::` einbinden.

## Code-Qualität (Rust)

### Fasslichkeit & Struktur
* Vermeide lange Prozeduren. Spalte sie in kleinere Prozeduren auf.
* Vermeide die Generierung von dupliziertem Code.
* Nutze Methoden (Traits wie `Result` | `Option` | `Iterator`), um die Schachtelung von `if` | `match` | `for` zu vermeiden und die Lesbarkeit zu erhöhen.

### Error-Handling & Robustheit
* Nutze konsequent `Result` und `Option` in Kombination mit dem `?`-Operator zur strukturierten Fehlerweiterleitung.
* Panics (`panic!`, `unwrap()`, `expect()`) sind im produktiven Code zu vermeiden.

## Sicherheit

### Secrets & Credentials
* Schreibe niemals Passwörter, API-Keys, Tokens, private Zertifikate oder vergleichbare Geheimnisse in den Quellcode, Kommentare, Beispiele, Test-Fixtures oder Dokumentation.
* Verwende für sensible Konfigurationswerte ausschließlich Umgebungsvariablen, dedizierte Secret-Manager oder externe Konfigurationsdateien, die via `.gitignore` ausgeschlossen sind.
* Auch scheinbar harmlose Beispielwerte (z. B. `password = "secret"`, `api_key = "12345"`) sind zu vermeiden, da sie versehentlich committet werden könnten. Nutze stattdessen generische Platzhalter wie `<API_KEY>`.
* Vor jeder Änderung ist zu prüfen, ob versehentlich Secrets in den Diff eingeflossen sind. Im Zweifel ist der Nutzer hinzuweisen, bevor committed wird.
