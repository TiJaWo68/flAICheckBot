package de.in.flaicheckbot.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;
import javax.swing.Scrollable;
import javax.swing.SwingConstants;

/**
 * Reusable panel for displaying images with zoom, pan, and selection/crop
 * capabilities.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class ZoomableImagePanel extends JPanel implements Scrollable {
    private BufferedImage originalImage;
    private BufferedImage currentImage;
    private double zoom = -1.0; // -1 = fit
    private Point startPoint;
    private Point endPoint;
    private Rectangle selection;

    public ZoomableImagePanel() {
        MouseAdapter adapter = new MouseAdapter() {
            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                if (e.isControlDown()) {
                    adjustZoom((e.getWheelRotation() < 0) ? .2f : -.2f);
                }
            }

            @Override
            public void mousePressed(MouseEvent e) {
                startPoint = e.getPoint();
                selection = null;
                repaint();
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                endPoint = e.getPoint();
                int x = Math.min(startPoint.x, endPoint.x);
                int y = Math.min(startPoint.y, endPoint.y);
                int width = Math.abs(startPoint.x - endPoint.x);
                int height = Math.abs(startPoint.y - endPoint.y);
                selection = new Rectangle(x, y, width, height);
                repaint();
            }
        };
        addMouseListener(adapter);
        addMouseMotionListener(adapter);

        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                if (zoom < 0) {
                    recalculateSize();
                }
            }
        });
    }

    public void setImage(BufferedImage img) {
        this.originalImage = img;
        this.currentImage = (img != null) ? deepCopy(img) : null;
        this.zoom = -1.0;
        this.selection = null;
        recalculateSize();
    }

    public void updateImage(BufferedImage img) {
        this.currentImage = deepCopy(img);
        this.selection = null;
        recalculateSize();
    }

    public BufferedImage getImage() {
        return currentImage;
    }

    public void reset() {
        if (originalImage != null) {
            currentImage = deepCopy(originalImage);
            selection = null;
            recalculateSize();
        }
    }

    public void fitToScreen() {
        this.zoom = -1.0;
        recalculateSize();
    }

    public void fitToWidth() {
        this.zoom = -2.0;
        recalculateSize();
    }

    public void adjustZoom(double delta) {
        if (currentImage == null)
            return;
        if (zoom < 0) {
            zoom = getCurrentScale();
        }
        zoom = Math.max(0.1, zoom + delta);
        recalculateSize();
    }

    public void cropSelection() {
        if (selection == null || currentImage == null)
            return;

        double scale = getCurrentScale();
        int imgX = (int) (selection.x / scale);
        int imgY = (int) (selection.y / scale);
        int imgW = (int) (selection.width / scale);
        int imgH = (int) (selection.height / scale);

        // Bounds check
        imgX = Math.max(0, Math.min(imgX, currentImage.getWidth() - 1));
        imgY = Math.max(0, Math.min(imgY, currentImage.getHeight() - 1));
        imgW = Math.min(imgW, currentImage.getWidth() - imgX);
        imgH = Math.min(imgH, currentImage.getHeight() - imgY);

        if (imgW > 0 && imgH > 0) {
            currentImage = deepCopy(currentImage.getSubimage(imgX, imgY, imgW, imgH));
            selection = null;
            recalculateSize();
        }
    }

    private double getCurrentScale() {
        if (currentImage == null)
            return 1.0;
        if (zoom < 0) {
            // Priority 1: Current panel width (if we are already in the layout)
            int targetWidth = getWidth();
            int targetHeight = getHeight();

            // Priority 2: Parent view width (e.g. JViewport)
            if (targetWidth <= 0 && getParent() != null) {
                targetWidth = getParent().getWidth();
                targetHeight = getParent().getHeight();
            }

            if (targetWidth <= 0)
                return 1.0;

            double zoomX = (double) targetWidth / currentImage.getWidth();
            double zoomY = (double) targetHeight / currentImage.getHeight();

            if (zoom == -1.0) {
                double s = Math.min(zoomX, zoomY);
                return (s > 1.0) ? 1.0 : s; // Don't upscale on fit
            } else if (zoom == -2.0) {
                return (zoomX > 1.0) ? 1.0 : zoomX;
            }
        }
        return zoom;
    }

    private void recalculateSize() {
        if (currentImage != null) {
            double s = getCurrentScale();
            Dimension d = new Dimension((int) (currentImage.getWidth() * s),
                    (int) (currentImage.getHeight() * s));
            setPreferredSize(d);
            revalidate();
        }
        repaint();
    }

    private BufferedImage deepCopy(BufferedImage source) {
        if (source == null)
            return null;
        int type = source.getType();
        if (type == BufferedImage.TYPE_CUSTOM || type == 0) {
            type = BufferedImage.TYPE_INT_ARGB;
        }
        BufferedImage b = new BufferedImage(source.getWidth(), source.getHeight(), type);
        Graphics g = b.getGraphics();
        g.drawImage(source, 0, 0, null);
        g.dispose();
        return b;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (currentImage != null) {
            double s = getCurrentScale();
            int w = (int) (currentImage.getWidth() * s);
            int h = (int) (currentImage.getHeight() * s);
            g.drawImage(currentImage, 0, 0, w, h, null);

            if (selection != null) {
                Graphics2D g2 = (Graphics2D) g;
                g2.setColor(new Color(0, 120, 215, 100));
                g2.fillRect(selection.x, selection.y, selection.width, selection.height);
                g2.setColor(Color.BLUE);
                g2.drawRect(selection.x, selection.y, selection.width, selection.height);
            }
        }
    }

    // --- Scrollable Implementation ---

    @Override
    public Dimension getPreferredScrollableViewportSize() {
        return getPreferredSize();
    }

    @Override
    public int getScrollableUnitIncrement(Rectangle visibleRect, int orientation, int direction) {
        return 16;
    }

    @Override
    public int getScrollableBlockIncrement(Rectangle visibleRect, int orientation, int direction) {
        return (orientation == SwingConstants.VERTICAL) ? visibleRect.height : visibleRect.width;
    }

    @Override
    public boolean getScrollableTracksViewportWidth() {
        return zoom == -2.0 || zoom == -1.0;
    }

    @Override
    public boolean getScrollableTracksViewportHeight() {
        return zoom == -1.0;
    }
}
