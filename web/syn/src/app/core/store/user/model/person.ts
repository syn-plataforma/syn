export interface BasePerson {
  identifier: string;
  name: string;
  surname1: string;
  surname2: string;
  fullName: string;
}

export interface Person extends BasePerson {
  representative: BasePerson|null;
}

export const initialPerson: Person = {
  identifier: '',
  name: '',
  surname1: '',
  surname2: '',
  fullName: '',
  representative: null,
};
