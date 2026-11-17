package de.in.fcb.i18n;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.Locale;

/**
 * Formats calendar dates according to the session locale.
 */
public final class DateFormatter {

    private DateFormatter() {
    }

    public static String format(LocalDate date, Locale locale) {
        if (date == null) {
            return "";
        }
        Locale effective = locale == null ? Locale.GERMANY : locale;
        return DateTimeFormatter.ofLocalizedDate(FormatStyle.MEDIUM).withLocale(effective).format(date);
    }
}
