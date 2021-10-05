/* eslint-disable @typescript-eslint/naming-convention */
export interface UserResponse {
  data: {
    id: number;
    email: string;
    first_name: string;
    last_name: string;
  };
  ad: {
    company: string;
    url: string;
    text: string;
  };
}

export const initialUserResponse: UserResponse = {
  data: {
    id: 0,
    email: '',
    first_name: '',
    last_name: '',
  },
  ad: {
    company: '',
    url: '',
    text: '',
  },
};
