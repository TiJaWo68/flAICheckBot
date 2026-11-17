package de.in.fcb.logging;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.InetSocketAddress;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import net.logstash.logback.appender.LogstashTcpSocketAppender;
import net.logstash.logback.appender.destination.DestinationParser;

class LoggingConfigurationTest {

    private static final int DEFAULT_PORT = 4560;

    @Test
    @DisplayName("Should parse single destination with host and port")
    void shouldParseSingleDestination() {
        List<InetSocketAddress> destinations = DestinationParser.parse("192.168.178.110:4560", DEFAULT_PORT);

        assertThat(destinations).hasSize(1);
        assertThat(destinations.get(0).getHostString()).isEqualTo("192.168.178.110");
        assertThat(destinations.get(0).getPort()).isEqualTo(4560);
    }

    @Test
    @DisplayName("Should parse multiple comma-separated destinations with ports for failover")
    void shouldParseMultipleDestinationsWithPorts() {
        List<InetSocketAddress> destinations = DestinationParser.parse(
                "192.168.178.110:4560,192.168.178.135:4560",
                DEFAULT_PORT);

        assertThat(destinations).hasSize(2);
        assertThat(destinations.get(0).getHostString()).isEqualTo("192.168.178.110");
        assertThat(destinations.get(0).getPort()).isEqualTo(4560);
        assertThat(destinations.get(1).getHostString()).isEqualTo("192.168.178.135");
        assertThat(destinations.get(1).getPort()).isEqualTo(4560);
    }

    @Test
    @DisplayName("Should apply default port when port is omitted in destination list")
    void shouldApplyDefaultPortWhenOmitted() {
        List<InetSocketAddress> destinations = DestinationParser.parse(
                "192.168.178.110,192.168.178.135",
                DEFAULT_PORT);

        assertThat(destinations).hasSize(2);
        assertThat(destinations.get(0).getHostString()).isEqualTo("192.168.178.110");
        assertThat(destinations.get(0).getPort()).isEqualTo(4560);
        assertThat(destinations.get(1).getHostString()).isEqualTo("192.168.178.135");
        assertThat(destinations.get(1).getPort()).isEqualTo(4560);
    }

    @Test
    @DisplayName("Appender should register multiple destinations correctly via addDestination")
    void appenderShouldRegisterMultipleDestinations() {
        LogstashTcpSocketAppender appender = new LogstashTcpSocketAppender();
        appender.addDestination("192.168.178.110:4560,192.168.178.135:4560");

        assertThat(appender.getDestinations()).hasSize(2);
        assertThat(appender.getDestinations().get(0).getHostString()).isEqualTo("192.168.178.110");
        assertThat(appender.getDestinations().get(1).getHostString()).isEqualTo("192.168.178.135");
    }
}
