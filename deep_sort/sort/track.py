# vim: expandtab:ts=4:sw=4
from .nn_matching import _cosine_distance

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, stored_embeddings,
                 matching_conf ,feature=None):
        #
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.face_data = stored_embeddings[0]
        self.name_list = stored_embeddings[1]
        self.matching_conf = matching_conf
        self.track_name = ""
        self.temp_name = ""
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.name_update = 0

        self.state = TrackState.Tentative

        # feature cache
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()  # xc,yc, a, h
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # 基于当前观测的结果，更新 KF 的mean variance 矩阵
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        # 把当前的观测 feature 加入 feature list
        self.features.append(detection.feature)
        # 这里的features 长度永远是1

        self.hits += 1
        self.time_since_update = 0


        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            
            self.state = TrackState.Confirmed


        if self.state == TrackState.Confirmed:
            dist_list=[]
            new_name = []

            for idx, emb_db in enumerate(self.face_data):
                dist = _cosine_distance(emb_db[0].reshape(1,-1), detection.feature.reshape(1,-1) )[0][0]
                dist = round(dist,3)
                # print("new nameeeeeeee.....",len(new_name))
                check = True
                if len(new_name)!=0:
                    for idx2, names in enumerate(new_name):
                        if names == self.name_list[idx]:
                            dist_list[idx2] = round((dist_list[idx2] + dist)/2.0,3)
                            check = False
                            break

                    if check:
                        new_name.append(self.name_list[idx])
                        dist_list.append(dist)



                else:
                    new_name.append(self.name_list[idx])
                    dist_list.append(dist)
                

                # print (self.name_list[idx], dist)
                # print(self.matching_conf)

            # for idx , names in enumerate(new_name):
            #     print(new_name[idx], dist_list[idx])
            idx_min=dist_list.index(min(dist_list))

            if dist_list[idx_min] <= self.matching_conf:
                if new_name[idx_min] != self.temp_name:
                    self.name_update=0
                    self.temp_name = new_name[idx_min]
                else:
                    self.name_update+=1
                    if self.name_update >= 3:
                        self.track_name = self.temp_name
                        self.name_update = 0

            else:
                self.name_update = 0
                self.track_name = ("Undetected")

        #_______ editing till here

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
