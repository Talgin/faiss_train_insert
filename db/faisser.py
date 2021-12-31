import psycopg2
import numpy as np
from shutil import copyfile
import faiss as fs
import os


class Faisser:
    def __init__(self, faiss_path):
        if not os.path.exists(faiss_path):
            message = {
                    'status': 'error',
                    'message': 'NO FAISS FILE FOUND, PLEASE CHECK LOCATION OF INDEX'
                    }
            print(message)
        else:
            self.faiss_index = fs.read_index(faiss_path, fs.IO_FLAG_ONDISK_SAME_DIR)
        self.faiss_path = faiss_path


    def get_records_amount(self):
        """Getting records amount stored in faiss index

        Returns
        -------
        amount : int
            amount of records in faiss index
        """
        amount = self.faiss_index.ntotal
        return amount


    def save_faiss_index(self, path_to_save):
        """Saving modified faiss index in the give location

        Parameters
        ----------
        path_to_save : str
            Path to the location where to save the file

        Returns
        -------
        boolean : bool
            if successful return True, else False
        """
        try:
            fs.write_index(self.faiss_index, path_to_save)
            return True
        except:
            return False


    def delete_from_faiss(self, ud_code):
        """Delete record from faiss index by id

        Parameters
        ----------
        ud_code : str
            Id of the current image obtained from the name of image

        Returns
        -------
        boolean : bool
            if successful returns True, else False

        """
        try:
            self.faiss_index.remove_ids(np.array([int(ud_code)]))
            return True
        except:
            return False


    def search_from_faiss_top_n(self, one_vector, top_n):
        """Search top 1 record from faiss index that matches given embedding

        Parameters
        ----------
        one_vector : np.array
            (1,512) array with feature embedding
        top_n : int
            integer in range 1 to size of index

        Returns
        -------
        distance, index : tuple
            tuple with distances and indices of records or (None, None) if not found
        """
        try:
            topn = 1
            if self.faiss_index.ntotal >= top_n:
                topn = top_n
            else:
                topn = self.faiss_index.ntotal

            if self.faiss_index.ntotal > 1000000:
                self.faiss_index.nprobe = 4096
            else:
                self.faiss_index.nprobe = 256

            query = np.array([one_vector], dtype=np.float32)
            D, I = self.faiss_index.search(query, topn)

            return D, I
        except:
            return None, None


    def search_from_faiss_top_1(self, one_vector, threshold):
        """Search top 1 record from faiss index that matches given embedding

        Parameters
        ----------
        one_vector : np.array
            (1,512) array with feature embedding
        threshold : int
            integer in range 0 to 100

        Returns
        -------
        distance, index : tuple
            tuple with distance and index of the record or (None, None) if not found
        """
        try:
            topn = 1
            if self.faiss_index.ntotal > 1000000:
                self.faiss_index.nprobe = 4096
            else:
                self.faiss_index.nprobe = 256
            query = np.array([one_vector], dtype=np.float32)
            D, I = self.faiss_index.search(query, topn)
            distance = float(threshold)/100

            if D[0][0] >= distance:
                return D[0], I[0]
            else:
                return None, None
        except:
            return None, None


    def insert_into_faiss(self, ud_code, feature):
        """Inserting id and embedding into faiss index

        Parameters
        ----------
        ud_code: str
            id of the current image
        feature : np.array
            (1,512) sized feature embedding of the image

        Returns
        -------
        boolean : bool
            if successfule returns True, else False
        """
        try:
            vector = np.array([feature], dtype=np.float32)
            ids = np.array([int(ud_code)])
            # self.faiss_index.train(vector)
            self.faiss_index.add_with_ids(vector, ids)
            return True
        except:
            return False


    def search_new_person(self, vector, distance):
        res = None
        try:
            vectors_archive = pd.read_sql('SELECT * FROM fr.vectors_archive', self.engine)
            # if this does not work, try to divide table into multiple dataframes
            unique_id = vectors_archive['unique_id']
            vectors = vectors_archive['vector']
            cameras = vectors_archive['camera_id']
            # servers = vectors_archive['server_id']
        except:
            print('Could not get data from fr.vectors_archive')
            return res
        distance = float(distance)/100
        try:
            res = {'person': []}
            for i in range(len(vectors)):
                vec = np.fromstring(vectors[i][1:-1], dtype=float, sep=',')
                dist = np.dot(vector, vec)
                if dist > distance:
                    dct = {'unique_id': unique_id[i], 'score': dist, 'camera': cameras[i]}
                    res['person'].append(dct)
                    copyfile('/home/dastrix/PROJECTS/face_reco_2_loc/face_reco_2_loc/application/static/frames_folder/' + str(unique_id[i]) + '_' + str(cameras[i]) + '.jpg', '/home/dastrix/PROJECTS/face_reco_2_loc/face_reco_2_loc/application/static/search_result/' + str(unique_id[i]) + '_' + str(cameras[i]) + '.jpg')
        except:
            print('Could not compare features')

        return res


    def insert_person_into_faiss(self, ud_code, feature):
        """Inserting id and embedding into faiss index

        Parameters
        ----------
        ud_code: str
            id of the current image
        feature : np.array
            (1,512) sized feature embedding of the image

        Returns
        -------
        message : dict
            dictionary with statuses
        """
        message = None
        current_faiss = self.get_records_amount()
        if current_faiss > 0:
            faiss_res = self.insert_into_faiss(ud_code, feature)
            if faiss_res:
                save = self.save_faiss_index(self.faiss_path)
                if save:
                    if self.get_records_amount() > current_faiss:
                        message = { 'status': 'success',
                                    'message': 'Record successfully inserted into INDEX',
                                    'ud_code': ud_code,
                                    'previous_amount': current_faiss,
                                    'updated_amount': self.get_records_amount()
                                    }
                    else:
                        message = {
                                'status': 'error',
                                'message': 'INDEX NOT CHANGED'
                                }
                else:
                    message = {
                    'status': 'error',
                    'message': 'COULD NOT SAVE INDEX'
                    }
                return message
        else:
            message = {
                    'status': 'error',
                    'message': 'INDEX IS EMPTY OR NOT PROPERLY READ'
                    }
            return message


    def delete_person_from_faiss(self, ud_code):
        """Delete person by id from faiss index

        Parameters
        ----------
        ud_code : str
            Id of the current image obtained from the name of image

        Returns
        -------
        message : dict
            dictionary with statuses

        """
        message = None
        previous = self.get_records_amount()
        # faiss_index.remove_ids(np.array([int(ud_code)]))
        delete = delete_from_faiss(ud_code)
        if delete:
            if self.get_records_amount() < previous:
                save = self.save_faiss_index(self.faiss_path)
                if save:
                    message = {
                                'status': 'success',
                                'message': 'Person deleted from INDEX',
                                'ud_code': ud_code,
                                'number of people in faiss': self.get_records_amount()
                                }
                else:
                    message = {
                                'status': 'error',
                                'message': 'Deletion not successful. Index not changed.'
                                }
                return message
            else:
                message = {
                            'status': 'error',
                            'message': 'Could not delete from FAISS index. Person is already deleted or does not exist.'
                            }
                return message
        else:
            message = {
                    'status': 'error',
                    'message': 'Deletion not successful. Index not changed.'
                    }
            return message


    def search_person_from_faiss(self, feature, threshold):
        """Test search of the person from faiss index

        Parameters
        ----------
        feature : np.array
            (1,512) array with feature embedding
        threshold : int
            integer in range 0 to 100

        Returns
        -------
        message : dict
            dictionary with statuses
        """
        message = None
        if len(feature) > 0:
            distance, index = self.search_from_faiss_top_1(feature, threshold)
            if index is not None:
                message = {
                        'status': 'success',
                        'distance': distance,
                        'ud_code': index,
                        'message': 'Record successfully found'
                        }
                return message
            else:
                message = {
                        'status': 'error',
                        'ud_code': None,
                        'message': 'No person found'
                        }
                return message
        else:
            message = {'status': 'error', 'message': 'wrong embedding'}
            return message
