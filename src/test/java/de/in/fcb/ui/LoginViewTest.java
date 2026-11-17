package de.in.fcb.ui;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.vaadin.flow.component.Component;
import com.vaadin.flow.component.html.H1;
import com.vaadin.flow.component.html.Image;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.BeforeEnterEvent;
import com.vaadin.flow.router.Location;
import com.vaadin.flow.router.QueryParameters;

import de.in.fcb.security.AppUserRepository;
import de.in.fcb.ui.setup.SetupView;

class LoginViewTest {

    private AppUserRepository repository;

    @BeforeEach
    void setUp() {
        repository = mock(AppUserRepository.class);
    }

    @Test
    void forwardsToSetupWhenNoUserExists() {
        when(repository.count()).thenReturn(0L);
        LoginView view = new LoginView(repository, "");
        BeforeEnterEvent event = mock(BeforeEnterEvent.class);

        view.beforeEnter(event);

        verify(event).forwardTo(SetupView.class);
    }

    @Test
    void doesNotForwardToSetupWhenAUserExists() {
        when(repository.count()).thenReturn(1L);
        LoginView view = new LoginView(repository, "");
        BeforeEnterEvent event = mock(BeforeEnterEvent.class);
        Location location = mock(Location.class);
        QueryParameters queryParameters = mock(QueryParameters.class);
        when(event.getLocation()).thenReturn(location);
        when(location.getQueryParameters()).thenReturn(queryParameters);
        when(queryParameters.getParameters()).thenReturn(Map.of());

        view.beforeEnter(event);

        verify(event, never()).forwardTo(SetupView.class);
    }

    @Test
    void hidesGoogleLoginWhenClientIdIsNotConfigured() {
        LoginView view = new LoginView(repository, "");

        assertThat(view.getChildren().count()).isEqualTo(2);
    }

    @Test
    void showsGoogleLoginWhenClientIdIsConfigured() {
        LoginView view = new LoginView(repository, "google-client-id");

        assertThat(view.getChildren().count()).isEqualTo(3);
    }

    @Test
    void rendersBrandHeaderAndOmitsRedundantFormTitle() {
        LoginView view = new LoginView(repository, "");

        List<Component> children = view.getChildren().toList();
        assertThat(children.get(0)).isInstanceOf(VerticalLayout.class);

        VerticalLayout header = (VerticalLayout) children.get(0);
        List<Component> headerItems = header.getChildren().toList();
        assertThat(headerItems).hasSize(3);
        assertThat(headerItems.get(0)).isInstanceOf(Image.class);
        assertThat(headerItems.get(1)).isInstanceOf(H1.class);
        assertThat(headerItems.get(2)).isInstanceOf(Span.class);

        Image logo = (Image) headerItems.get(0);
        assertThat(logo.getSrc()).isEqualTo("/icons/flaicheckbot-logo.svg");

        assertThat(view.createLoginI18n().getForm().getTitle()).isEmpty();
    }
}
