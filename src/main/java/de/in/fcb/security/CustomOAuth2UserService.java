package de.in.fcb.security;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.OAuth2Error;
import org.springframework.security.oauth2.core.OAuth2ErrorCodes;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;

/**
 * Loads OAuth2 users (e.g. Google) and maps them onto pre-existing local
 * {@code app_user} accounts so authorization stays in the database.
 */
@Service
public class CustomOAuth2UserService extends DefaultOAuth2UserService {

    private static final String SUBJECT_ATTRIBUTE = "sub";
    private static final String EMAIL_ATTRIBUTE = "email";

    private final AppUserRepository appUserRepository;

    public CustomOAuth2UserService(AppUserRepository appUserRepository) {
        this.appUserRepository = appUserRepository;
    }

    @Override
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {
        return buildOAuth2User(super.loadUser(userRequest));
    }

    OAuth2User buildOAuth2User(OAuth2User oAuth2User) {
        String providerId = oAuth2User.getAttribute(SUBJECT_ATTRIBUTE);
        String email = oAuth2User.getAttribute(EMAIL_ATTRIBUTE);

        AppUser appUser = findWhitelistedUser(providerId, email)
                .orElseThrow(() -> new OAuth2AuthenticationException(new OAuth2Error(
                        OAuth2ErrorCodes.ACCESS_DENIED, "No enabled account for email: " + email, null)));

        if (!appUser.isEnabled()) {
            throw new OAuth2AuthenticationException(new OAuth2Error(
                    OAuth2ErrorCodes.ACCESS_DENIED, "Account is disabled", null));
        }

        List<GrantedAuthority> authorities = appUser.getRoles().stream()
                .map(role -> new SimpleGrantedAuthority(role.name()))
                .sorted(Comparator.comparing(GrantedAuthority::getAuthority))
                .collect(Collectors.toList());
        return new DefaultOAuth2User(authorities, oAuth2User.getAttributes(), SUBJECT_ATTRIBUTE);
    }

    private Optional<AppUser> findWhitelistedUser(String providerId, String email) {
        if (providerId != null) {
            Optional<AppUser> byProviderId = appUserRepository.findByProviderId(providerId);
            if (byProviderId.isPresent()) {
                return byProviderId;
            }
        }

        if (email == null || email.isBlank()) {
            return Optional.empty();
        }
        return appUserRepository.findByEmail(email);
    }
}
