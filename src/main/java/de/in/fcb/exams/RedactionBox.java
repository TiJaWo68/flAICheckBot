package de.in.fcb.exams;

/**
 * Normalized ({@code 0.0..1.0}) black redaction rectangle on a scan page.
 */
public record RedactionBox(double x, double y, double width, double height) {

    public RedactionBox rotateClockwise(int degreesClockwise) {
        int steps = (((degreesClockwise % 360) + 360) % 360) / 90;
        double rx = x;
        double ry = y;
        double rw = width;
        double rh = height;
        for (int i = 0; i < steps; i++) {
            double nextX = clamp01(1.0 - ry - rh);
            double nextY = clamp01(rx);
            double nextW = rh;
            double nextH = rw;
            rx = nextX;
            ry = nextY;
            rw = nextW;
            rh = nextH;
        }
        return new RedactionBox(rx, ry, rw, rh);
    }

    private static double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}
