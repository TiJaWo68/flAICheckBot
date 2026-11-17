package de.in.fcb.ui.setup;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.vaadin.flow.router.BeforeEnterEvent;

import de.in.fcb.security.AppUserRepository;
import de.in.fcb.security.SetupService;
import de.in.fcb.ui.LoginView;

class SetupViewTest {

    private AppUserRepository repository;
    private SetupService setupService;
    private BeforeEnterEvent event;

    @BeforeEach
    void setUp() {
        repository = mock(AppUserRepository.class);
        setupService = mock(SetupService.class);
        event = mock(BeforeEnterEvent.class);
    }

    @Test
    void forwardsToLoginWhenAUserAlreadyExists() {
        when(repository.count()).thenReturn(1L);
        SetupView view = new SetupView(repository, setupService);

        view.beforeEnter(event);

        verify(event).forwardTo(LoginView.class);
    }

    @Test
    void keepsSetupAvailableWhileNoUserExists() {
        when(repository.count()).thenReturn(0L);
        SetupView view = new SetupView(repository, setupService);

        view.beforeEnter(event);

        verify(event, never()).forwardTo(LoginView.class);
    }
}
