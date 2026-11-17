package de.in.fcb.admin;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

/**
 * Spring Data repository for {@link AppSetting}.
 */
@Repository
public interface AppSettingRepository extends JpaRepository<AppSetting, String> {

    Optional<AppSetting> findBySettingKey(String settingKey);
}
