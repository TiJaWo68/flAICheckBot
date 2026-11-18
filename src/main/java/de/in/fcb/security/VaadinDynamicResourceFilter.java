package de.in.fcb.security;

import java.io.IOException;

import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletRequestWrapper;
import jakarta.servlet.http.HttpServletResponse;

/**
 * Normalizes nested-route Vaadin dynamic resource paths (for example
 * {@code /evaluation/VAADIN/dynamic/resource/.../upload} produced after client-side
 * navigation to a parameterized route) back to {@code /VAADIN/dynamic/resource/...}
 * before Spring Security and Vaadin's {@code StreamRequestHandler} inspect the request.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class VaadinDynamicResourceFilter extends OncePerRequestFilter {

    private static final String DYNAMIC_RESOURCE_SEGMENT = "/VAADIN/dynamic/resource/";

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {
        String contextPath = request.getContextPath() == null ? "" : request.getContextPath();
        String uri = request.getRequestURI();
        if (uri != null && uri.startsWith(contextPath)) {
            String path = uri.substring(contextPath.length());
            int marker = path.indexOf(DYNAMIC_RESOURCE_SEGMENT);
            if (marker > 0) {
                String normalizedPath = path.substring(marker);
                filterChain.doFilter(new NormalizedResourceRequest(request, contextPath, normalizedPath), response);
                return;
            }
        }
        filterChain.doFilter(request, response);
    }

    private static final class NormalizedResourceRequest extends HttpServletRequestWrapper {

        private final String normalizedUri;
        private final String normalizedPath;

        NormalizedResourceRequest(HttpServletRequest request, String contextPath, String normalizedPath) {
            super(request);
            this.normalizedUri = contextPath + normalizedPath;
            this.normalizedPath = normalizedPath;
        }

        @Override
        public String getRequestURI() {
            return normalizedUri;
        }

        @Override
        public String getServletPath() {
            String original = super.getServletPath();
            return (original != null && !original.isEmpty()) ? normalizedPath : "";
        }

        @Override
        public String getPathInfo() {
            String original = super.getPathInfo();
            return original != null ? normalizedPath : null;
        }
    }
}
