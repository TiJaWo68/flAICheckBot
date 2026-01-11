package de.in.flaicheckbot.utils;

public class BackendCompatibilityChecker {

    public static boolean isCompatible(String appRequired, String backendActual) {
        if (backendActual == null || appRequired == null) {
            return false;
        }

        try {
            String[] reqParts = appRequired.split("\\.");
            String[] actParts = backendActual.split("\\.");

            if (reqParts.length != 3 || actParts.length != 3) {
                // Keep it simple: strict x.y.z format expected
                System.err.println("Version format mismatch: request=" + appRequired + ", actual=" + backendActual);
                return false;
            }

            int reqMajor = Integer.parseInt(reqParts[0]);
            int reqMinor = Integer.parseInt(reqParts[1]);
            // reqPatch is ignored for compatibility check usually, but let's parse it

            int actMajor = Integer.parseInt(actParts[0]);
            int actMinor = Integer.parseInt(actParts[1]);

            // Rule 1: Major must match exactly
            if (reqMajor != actMajor) {
                return false;
            }

            // Rule 2: Minor must be >= required
            if (actMinor < reqMinor) {
                return false;
            }

            // Rule 3: Patch is ignored (as per requirement)
            return true;

        } catch (NumberFormatException e) {
            System.err.println("Error parsing version numbers: " + e.getMessage());
            return false;
        }
    }
}
