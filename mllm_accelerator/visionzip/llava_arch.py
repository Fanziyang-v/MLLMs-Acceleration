def encode_images(self, images):
    image_features, keep_indices = self.get_model().get_vision_tower()(images)
    # image_features = self.get_model().vision_resampler(image_features, images=images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features
