package de.in.fcb.ui.setup;

import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.html.H1;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.EmailField;
import com.vaadin.flow.component.textfield.PasswordField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.data.binder.Binder;
import com.vaadin.flow.data.validator.EmailValidator;
import com.vaadin.flow.router.BeforeEnterEvent;
import com.vaadin.flow.router.BeforeEnterObserver;
import com.vaadin.flow.router.HasDynamicTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.server.auth.AnonymousAllowed;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.security.AppUserRepository;
import de.in.fcb.security.InvalidSetupTokenException;
import de.in.fcb.security.SetupService;
import de.in.fcb.ui.LoginView;

/**
 * Initial setup view that creates the first administrator account.
 * <p>
 * Permanently forwards to {@link LoginView} as soon as at least one account
 * exists.
 */
@Route("setup")
@AnonymousAllowed
public class SetupView extends VerticalLayout implements BeforeEnterObserver, HasDynamicTitle {

    private static final int MIN_PASSWORD_LENGTH = 8;

    private final AppUserRepository appUserRepository;
    private final SetupService setupService;

    private final TextField username = new TextField();
    private final EmailField email = new EmailField();
    private final PasswordField password = new PasswordField();
    private final PasswordField passwordConfirmation = new PasswordField();
    private final TextField setupToken = new TextField();
    private final Button submitButton = new Button();

    private final Binder<SetupForm> binder = new Binder<>(SetupForm.class);

    public SetupView(AppUserRepository appUserRepository, SetupService setupService) {
        this.appUserRepository = appUserRepository;
        this.setupService = setupService;

        setSizeFull();
        setJustifyContentMode(JustifyContentMode.CENTER);
        setAlignItems(Alignment.CENTER);

        configureFields();
        configureBinder();

        submitButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        submitButton.addClickListener(event -> createAdmin());

        FormLayout formLayout = new FormLayout();
        formLayout.add(username, email, password, passwordConfirmation, setupToken);
        formLayout.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));

        VerticalLayout card = new VerticalLayout();
        card.setMaxWidth("420px");
        card.setPadding(true);
        card.setAlignItems(Alignment.CENTER);
        card.addClassNames(
                LumoUtility.Background.CONTRAST_5,
                LumoUtility.BorderRadius.LARGE,
                LumoUtility.BoxShadow.MEDIUM,
                LumoUtility.Padding.XLARGE);

        H1 title = new H1(getTranslation("ui.setup.title"));
        title.addClassNames(LumoUtility.Margin.Bottom.XSMALL);

        card.add(title, formLayout, submitButton);
        add(card);
    }

    @Override
    public void beforeEnter(BeforeEnterEvent event) {
        if (appUserRepository.count() > 0) {
            event.forwardTo(LoginView.class);
        }
    }

    @Override
    public String getPageTitle() {
        return getTranslation("ui.setup.page-title");
    }

    private void configureFields() {
        username.setLabel(getTranslation("ui.setup.username.label"));
        email.setLabel(getTranslation("ui.setup.email.label"));
        password.setLabel(getTranslation("ui.setup.password.label"));
        passwordConfirmation.setLabel(getTranslation("ui.setup.password-confirmation.label"));
        setupToken.setLabel(getTranslation("ui.setup.token.label"));
        submitButton.setText(getTranslation("ui.setup.submit"));
    }

    private void configureBinder() {
        binder.forField(username)
                .asRequired(getTranslation("ui.setup.error.username-required"))
                .bind(SetupForm::getUsername, SetupForm::setUsername);

        binder.forField(email)
                .asRequired(getTranslation("ui.setup.error.email-required"))
                .withValidator(new EmailValidator(getTranslation("ui.setup.error.email-invalid")))
                .bind(SetupForm::getEmail, SetupForm::setEmail);

        binder.forField(password)
                .asRequired(getTranslation("ui.setup.error.password-required"))
                .withValidator(value -> value.length() >= MIN_PASSWORD_LENGTH,
                        getTranslation("ui.setup.error.password-min-length", MIN_PASSWORD_LENGTH))
                .bind(SetupForm::getPassword, SetupForm::setPassword);

        binder.forField(passwordConfirmation)
                .asRequired(getTranslation("ui.setup.error.password-confirmation-required"))
                .withValidator(value -> value.equals(password.getValue()),
                        getTranslation("ui.setup.error.password-mismatch"))
                .bind(SetupForm::getPasswordConfirmation, SetupForm::setPasswordConfirmation);

        binder.forField(setupToken)
                .asRequired(getTranslation("ui.setup.error.setup-token-required"))
                .bind(SetupForm::getSetupToken, SetupForm::setSetupToken);
    }

    private void createAdmin() {
        SetupForm form = new SetupForm();
        if (!binder.writeBeanIfValid(form)) {
            return;
        }

        try {
            setupService.createInitialAdmin(form.getUsername(), form.getEmail(), form.getPassword(),
                    form.getSetupToken());
            Notification.show(getTranslation("ui.setup.success.message"), 3000, Notification.Position.MIDDLE);
            getUI().ifPresent(ui -> ui.navigate(LoginView.class));
        } catch (InvalidSetupTokenException exception) {
            Notification.show(getTranslation("setup.error.invalid_token"), 5000, Notification.Position.MIDDLE);
        } catch (IllegalStateException exception) {
            Notification.show(getTranslation("ui.setup.error.already-configured"), 5000, Notification.Position.MIDDLE);
            getUI().ifPresent(ui -> ui.navigate(LoginView.class));
        }
    }

    /**
     * Form backing bean for the setup binder.
     */
    public static class SetupForm {

        private String username;
        private String email;
        private String password;
        private String passwordConfirmation;
        private String setupToken;

        public String getUsername() {
            return username;
        }

        public void setUsername(String username) {
            this.username = username;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }

        public String getPassword() {
            return password;
        }

        public void setPassword(String password) {
            this.password = password;
        }

        public String getPasswordConfirmation() {
            return passwordConfirmation;
        }

        public void setPasswordConfirmation(String passwordConfirmation) {
            this.passwordConfirmation = passwordConfirmation;
        }

        public String getSetupToken() {
            return setupToken;
        }

        public void setSetupToken(String setupToken) {
            this.setupToken = setupToken;
        }
    }
}
